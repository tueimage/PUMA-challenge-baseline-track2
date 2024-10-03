import numpy as np
import cv2
import zarr
import gc
import json
import os
import time
from skimage.segmentation import watershed
from scipy.ndimage import find_objects
from numcodecs import Blosc
from skimage.measure import regionprops
from src.constants import (
    MIN_THRESHS,
    MAX_THRESHS,
    MAX_HOLE_SIZE,
    LUT_MAGNIFICATION_MPP,
    LUT_MAGNIFICATION_X,
)
from src.data_utils import center_crop, NpyDataset, ImageDataset


def update_dicts(pinst_, pcls_, pcls_out, t_, old_ids, initial_ids):
    props = [(p.label, p.centroid) for p in regionprops(pinst_)]
    pcls_new = {}
    for id_, cen in props:
        try:
            pcls_new[str(id_)] = (pcls_[str(id_)], (cen[0] + t_[2], cen[1] + t_[0]))
        except KeyError:
            pcls_new[str(id_)] = (pcls_out[str(id_)], (cen[0] + t_[2], cen[1] + t_[0]))

    new_ids = [p[0] for p in props]

    for i in np.setdiff1d(old_ids, new_ids):
        try:
            del pcls_out[str(i)]
        except KeyError:
            pass
    for i in np.setdiff1d(new_ids, initial_ids):
        try:
            del pcls_new[str(i)]
        except KeyError:
            pass
    return pcls_out | pcls_new


def write(pinst_out, pcls_out, running_max, res, params):
    pinst_, pcls_, max_, t_, skip = res
    if not skip:
        if params["input_type"] != "wsi":
            pinst_.vindex[pinst_[:] != 0] += running_max
            pcls_ = {str(int(k) + running_max): v for k, v in pcls_.items()}
            props = [(p.label, p.centroid) for p in regionprops(pinst_)]
            pcls_new = {}
            for id_, cen in props:
                pcls_new[str(id_)] = (pcls_[str(id_)], (t_[-1], cen[0], cen[1]))

            running_max += max_
            pcls_out |= pcls_new
            pinst_out[t_[-1]] = np.asarray(pinst_, dtype=np.int32)

    return pinst_out, pcls_out, running_max


def work(tcrd, ds_coord, z, params):
    out_img = gen_tile_map(
        tcrd,
        ds_coord,
        params["ccrop"],
        model_out_p=params["model_out_p"],
        which="_inst",
        dim=params["out_img_shape"][-3],
        z=z,
        npy=params["input_type"] != "wsi",
    )
    out_cls = gen_tile_map(
        tcrd,
        ds_coord,
        params["ccrop"],
        model_out_p=params["model_out_p"],
        which="_cls",
        dim=params["out_cls_shape"][-3],
        z=z,
        npy=params["input_type"] != "wsi",
    )
    if params["input_type"] != "wsi":
        out_img = out_img[
            :,
            params["tile_size"] : -params["tile_size"],
            params["tile_size"] : -params["tile_size"],
        ]
        out_cls = out_cls[
            :,
            params["tile_size"] : -params["tile_size"],
            params["tile_size"] : -params["tile_size"],
        ]
    best_min_threshs = MIN_THRESHS
    best_max_threshs = MAX_THRESHS

    # using apply_func to apply along axis for npy stacks
    pred_inst, skip = faster_instance_seg(
        out_img, out_cls, params["best_fg_thresh_cl"], params["best_seed_thresh_cl"]
    )
    del out_img
    gc.collect()
    max_hole_size = MAX_HOLE_SIZE // 4
    if skip:
        pred_inst = zarr.array(
            pred_inst, compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        )

        return (pred_inst, {}, 0, tcrd, skip)
    pred_inst = post_proc_inst(
        pred_inst,
        max_hole_size,
    )
    pred_ct = make_ct(out_cls, pred_inst)
    del out_cls
    gc.collect()

    processed = remove_obj_cls(pred_inst, pred_ct, best_min_threshs, best_max_threshs)
    # TODO why is this here?
    pred_inst, pred_ct = processed
    max_inst = np.max(pred_inst)
    pred_inst = zarr.array(
        pred_inst.astype(np.int32),
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
    )
    return (pred_inst, pred_ct, max_inst, tcrd, skip)


def get_overlap_regions(tcrd, pad_size, out_img_shape):
    top = [tcrd[0], tcrd[0] + 2 * pad_size, tcrd[2], tcrd[3]] if tcrd[0] != 0 else None
    bottom = (
        [tcrd[1] - 2 * pad_size, tcrd[1], tcrd[2], tcrd[3]]
        if tcrd[1] != out_img_shape[-2]
        else None
    )
    left = [tcrd[0], tcrd[1], tcrd[2], tcrd[2] + 2 * pad_size] if tcrd[2] != 0 else None
    right = (
        [tcrd[0], tcrd[1], tcrd[3] - 2 * pad_size, tcrd[3]]
        if tcrd[3] != out_img_shape[-1]
        else None
    )
    d_top = [0, 2 * pad_size, 0, tcrd[3] - tcrd[2]]
    d_bottom = [
        tcrd[1] - tcrd[0] - 2 * pad_size,
        tcrd[1] - tcrd[0],
        0,
        tcrd[3] - tcrd[2],
    ]
    d_left = [0, tcrd[1] - tcrd[0], 0, 2 * pad_size]
    d_right = [
        0,
        tcrd[1] - tcrd[0],
        tcrd[3] - tcrd[2] - 2 * pad_size,
        tcrd[3] - tcrd[2],
    ]
    return (
        [top, bottom, left, right],
        [d_top, d_bottom, d_left, d_right],
        ["top", "bottom", "left", "right"],
    )  #


def get_subregions(which, shape):
    """
    Note that the names are incorrect :), inconsistency to be fixed with coordinates and xy swap
    """
    if which == "top":
        return [0, shape[0], 0, shape[1] // 4], [0, shape[0], 0, shape[1] // 2]
    elif which == "bottom":
        return [0, shape[0], (shape[1] * 3) // 4, shape[1]], [
            0,
            shape[0],
            shape[1] // 2,
            shape[1],
        ]
    elif which == "left":
        return [0, shape[0] // 4, 0, shape[1]], [0, shape[0] // 2, 0, shape[1]]
    elif which == "right":
        return [(shape[0] * 3) // 4, shape[0], 0, shape[1]], [
            shape[0] // 2,
            shape[0],
            0,
            shape[1],
        ]

    else:
        raise ValueError("Invalid which")


def expand_bbox(bbox, pad_size, img_size):
    return [
        max(0, bbox[0] - pad_size),
        max(0, bbox[1] - pad_size),
        min(img_size[0], bbox[2] + pad_size),
        min(img_size[1], bbox[3] + pad_size),
    ]


def get_tile_coords(shape, splits, pad_size, npy):
    if npy:
        tile_crds = [[0, shape[-2], 0, shape[-1], i] for i in range(shape[0])]
        return tile_crds

    else:
        shape = shape[-2:]
        tile_crds = []
        ts_1 = np.array_split(np.arange(0, shape[0]), splits)
        ts_2 = np.array_split(np.arange(0, shape[1]), splits)
        for i in ts_1:
            for j in ts_2:
                x_start = 0 if i[0] < pad_size else i[0] - pad_size
                x_end = shape[0] if i[-1] + pad_size > shape[0] else i[-1] + pad_size
                y_start = 0 if j[0] < pad_size else j[0] - pad_size
                y_end = shape[1] if j[-1] + pad_size > shape[1] else j[-1] + pad_size
                tile_crds.append([x_start, x_end, y_start, y_end])
    return tile_crds


def proc_tile(t, ccrop, which="_cls"):
    t = center_crop(t, ccrop, ccrop)
    if which == "_cls":
        t = t[1:]
        t = t.reshape(t.shape[0], -1)
        out = np.zeros(t.shape, dtype=bool)
        out[t.argmax(axis=0), np.arange(t.shape[1])] = 1
        t = out.reshape(-1, ccrop, ccrop)

    else:
        t = t[:2].astype(np.float16)
    return t


def gen_tile_map(
    tile_crd,
    ds_coord,
    ccrop,
    model_out_p="",
    which="_cls",
    dim=5,
    z=None,
    npy=False,
):
    if z is None:
        z = zarr.open(model_out_p + f"{which}.zip", mode="r")
    else:
        if which == "_cls":
            z = z[1]
        else:
            z = z[0]
    cadj = (z.shape[-1] - ccrop) // 2
    tx, ty, tz = None, None, None
    dtype = bool if which == "_cls" else np.float16

    if npy:
        # TODO fix npy
        coord_filter = ds_coord[:, 0] == tile_crd[-1]
        ds_coord_subset = ds_coord[coord_filter]
        zero_map = np.zeros(
            (dim, tile_crd[1] - tile_crd[0], tile_crd[3] - tile_crd[2]), dtype=dtype
        )
    else:
        zero_map = np.zeros(
            (dim, tile_crd[3] - tile_crd[2], tile_crd[1] - tile_crd[0]), dtype=dtype
        )
        coord_filter = (
            ((ds_coord[:, 0]) < tile_crd[1])
            & ((ds_coord[:, 0] + ccrop) > tile_crd[0])
            & ((ds_coord[:, 1]) < tile_crd[3])
            & ((ds_coord[:, 1] + ccrop) > tile_crd[2])
        )
        ds_coord_subset = ds_coord[coord_filter] - np.array([tile_crd[0], tile_crd[2]])

    z_address = np.arange(ds_coord.shape[0])[coord_filter]
    for _, (crd, tile) in enumerate(zip(ds_coord_subset, z[z_address])):
        if npy:
            tz, ty, tx = crd
        else:
            tx, ty = crd
        tx = tx
        ty = ty
        p_shift = [abs(i) if i < 0 else 0 for i in [ty, tx]]
        n_shift = [
            crd - (i + ccrop) if (i + ccrop) > crd else 0
            for i, crd in zip([ty, tx], zero_map.shape[1:3])
        ]
        try:
            zero_map[
                :,
                ty + p_shift[0] : ty + ccrop + n_shift[0],
                tx + p_shift[1] : tx + ccrop + n_shift[1],
            ] = proc_tile(tile, ccrop, which)[
                ...,
                p_shift[0] : ccrop + n_shift[0],
                p_shift[1] : ccrop + n_shift[1],
            ]

        except:
            print(zero_map.shape)
            print(tx)
            print(ty)
            print(ccrop)
            print(tile.shape)
            raise ValueError
    return zero_map


def faster_instance_seg(out_img, out_cls, best_fg_thresh_cl, best_seed_thresh_cl):
    _, rois = cv2.connectedComponents((out_img[0] > 0).astype(np.uint8), connectivity=8)
    bboxes = find_objects(rois)
    del rois
    gc.collect()
    skip = False
    labelling = zarr.zeros(
        out_cls.shape[1:],
        dtype=np.int32,
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    if len(bboxes) == 0:
        skip = True
        return labelling, skip
    max_inst = 0
    for bb in bboxes:
        bg_pred = out_img[(slice(0, 1, None), *bb)].squeeze()
        if (
            (np.array(bg_pred.shape[-2:]) <= 2).any()
            | (np.array(bg_pred.shape).sum() <= 64)
            | (len(bg_pred.shape) < 2)
        ):
            continue
        fg_pred = out_img[(slice(1, 2, None), *bb)].squeeze()
        sem = out_cls[(slice(0, len(best_fg_thresh_cl), None), *bb)]
        ws_surface = 1.0 - fg_pred  # .astype(np.float32)
        fg = np.zeros_like(ws_surface, dtype="bool")
        seeds = np.zeros_like(ws_surface, dtype="bool")

        for cl, fg_t in enumerate(best_fg_thresh_cl):
            mask = sem[cl]
            fg[mask] |= (1.0 - bg_pred[mask]) > fg_t
            seeds[mask] |= fg_pred[mask] > best_seed_thresh_cl[cl]

        del fg_pred, bg_pred, sem, mask
        gc.collect()
        _, markers = cv2.connectedComponents((seeds).astype(np.uint8), connectivity=8)
        del seeds
        gc.collect()
        bb_ws = watershed(ws_surface, markers, mask=fg, connectivity=2)
        del ws_surface, markers, fg
        gc.collect()
        bb_ws[bb_ws != 0] += max_inst
        labelling[bb] = bb_ws
        max_inst = np.max(bb_ws)
        del bb_ws
        gc.collect()
    return labelling, skip


def post_proc_inst(
    pred_inst,
    hole_size=50,
):
    pshp = pred_inst.shape
    pred_inst = np.asarray(pred_inst)
    init = find_objects(pred_inst)
    init_large = []
    adj = 8
    for i, sl in enumerate(init):
        if sl:
            slx1 = sl[0].start - adj if (sl[0].start - adj) > 0 else 0
            slx2 = sl[0].stop + adj if (sl[0].stop + adj) < pshp[0] else pshp[0]
            sly1 = sl[1].start - adj if (sl[1].start - adj) > 0 else 0
            sly2 = sl[1].stop + adj if (sl[1].stop + adj) < pshp[1] else pshp[1]
            init_large.append(
                (i + 1, (slice(slx1, slx2, None), slice(sly1, sly2, None)))
            )
    out = np.zeros(pshp, dtype=np.int32)
    i = 1
    for sl in init_large:
        rm_small_hole = remove_small_holescv2(pred_inst[sl[1]] == (sl[0]), hole_size)
        out[sl[1]][rm_small_hole > 0] = i
        i += 1

    del pred_inst
    gc.collect()

    after_sh = find_objects(out)
    out_ = np.zeros(out.shape, dtype=np.int32)
    i_ = 1
    for i, sl in enumerate(after_sh):
        i += 1
        if sl:
            nr_objects, relabeled = cv2.connectedComponents(
                (out[sl] == i).astype(np.uint8), connectivity=8
            )
            for new_lab in range(1, nr_objects):
                out_[sl] += (relabeled == new_lab) * i_
                i_ += 1
    return out_


def make_ct(pred_class, instance_map):
    if type(pred_class) != np.ndarray:
        pred_class = pred_class[:]
    slices = find_objects(instance_map)
    pred_class = np.rollaxis(pred_class, 0, 3)
    # pred_class = softmax(pred_class,0)
    out = []
    out.append((0, 0))
    for i, sl in enumerate(slices):
        i += 1
        if sl:
            inst = instance_map[sl] == i
            i_cls = pred_class[sl][inst]
            i_cls = np.sum(i_cls, axis=0).argmax() + 1
            out.append((i, i_cls))
    out_ = np.array(out)
    pred_ct = {str(k): int(v) for k, v in out_ if v != 0}
    return pred_ct


def remove_obj_cls(pred_inst, pred_cls_dict, best_min_threshs, best_max_threshs):
    out_oi = np.zeros_like(pred_inst, dtype=np.int64)
    i_ = 1
    out_oc = []
    out_oc.append((0, 0))
    slices = find_objects(pred_inst)

    for i, sl in enumerate(slices):
        i += 1
        px = np.sum([pred_inst[sl] == i])
        cls_ = pred_cls_dict[str(i)]
        if (px > best_min_threshs[cls_ - 1]) & (px < best_max_threshs[cls_ - 1]):
            out_oc.append((i_, cls_))
            out_oi[sl][pred_inst[sl] == i] = i_
            i_ += 1
    out_oc = np.array(out_oc)
    out_dict = {str(k): int(v) for k, v in out_oc if v != 0}
    return out_oi, out_dict


def remove_small_holescv2(img, sz):
    # this is still pretty slow but at least its a bit faster than other approaches?
    img = np.logical_not(img).astype(np.uint8)

    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[1:, -1]
    nb_blobs -= 1
    im_result = np.zeros((img.shape), dtype=np.uint16)
    for blob in range(nb_blobs):
        if sizes[blob] >= sz:
            im_result[im_with_separated_blobs == blob + 1] = 1

    im_result = np.logical_not(im_result)
    return im_result


def get_pp_params(params, mit_eval=False):
    eval_metric = "f1"
    fg_threshs = []
    seed_threshs = []
    exp = params["data_dir"]
    mod_path = os.path.join(params["root"], exp)
    with open(os.path.join(mod_path, "param_dict.json"), "r") as js:
            dt = json.load(js)
            fg_threshs.append(dt[f"best_fg_{eval_metric}"])
            seed_threshs.append(dt[f"best_seed_{eval_metric}"])
    params["best_fg_thresh_cl"] = np.mean(fg_threshs, axis=0)
    params["best_seed_thresh_cl"] = np.mean(seed_threshs, axis=0)
    print(params["best_fg_thresh_cl"], params["best_seed_thresh_cl"])

    return params


def get_shapes(params, nclasses):
    padding_factor = params["overlap"]
    tile_size = params["tile_size"]
    ds_factor = 1
    if params["input_type"] in ["img", "npy"]:
        if params["input_type"] == "npy":
            dataset = NpyDataset(
                params["p"],
                tile_size,
                padding_factor=padding_factor,
                ratio_object_thresh=0.3,
                min_tiss=0.1,
            )
        else:
            dataset = ImageDataset(
                params["p"],
                params["tile_size"],
                padding_factor=params["overlap"],
                ratio_object_thresh=0.3,
                min_tiss=0.1,
            )
        params["orig_shape"] = dataset.orig_shape[:-1]
        ds_coord = np.array(dataset.idx).astype(int)
        shp = dataset.store.shape

        ccrop = int(dataset.padding_factor * dataset.crop_size_px)
        coord_adj = (dataset.crop_size_px - ccrop) // 2
        ds_coord[:, 1:] += coord_adj
        out_img_shape = (shp[0], 2, shp[1], shp[2])
        out_cls_shape = (shp[0], nclasses, shp[1], shp[2])
    else:
        print("ERROR: Input type is not an image")

    params["ds_factor"] = ds_factor
    params["out_img_shape"] = out_img_shape
    params["out_cls_shape"] = out_cls_shape
    params["ccrop"] = ccrop

    return params, ds_coord
