import os
import argparse
import shutil
import sys
from timeit import default_timer as timer
from datetime import timedelta
import torch
from src.inference import inference_main, get_inference_setup
from src.post_process import post_process_main
from src.data_utils import copy_img

torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count(), " cuda devices")


def main(params: dict):
    """
    Start nuclei segmentation and classification pipeline using specified parameters from argparse

    Parameters
    ----------
    params: dict
        input parameters from argparse
    """

    print("Optimizing postprocessing for F1-score")
    params["root"] = os.path.dirname(__file__)
    params["data_dir"] = params["root"] + params["cp"]

    print("saving results to:", params["output"])
    print("loading model from:", params["data_dir"])

    # Run per tile inference and store results
    params, models, augmenter, color_aug_fn = get_inference_setup(params)

    # pick the first (and only) file in folder
    input_filename = os.path.join(params["input"], os.listdir(params["input"])[0])
    # input_filename = os.path.join(params["root"] + params["input"], os.listdir(params["root"] + params["input"])[0])
    print(f"Running inference on {input_filename}")

    start_time = timer()
    params["p"] = input_filename.rstrip()
    params["ext"] = os.path.splitext(params["p"])[-1]
    if params["ext"] != ".tif":
        print("ERROR: input type is not a .tif file")
    params["input_type"] = "img"
    print("Processing ", params["p"])
    if params["cache"] is not None:
        print("Caching input at:")
        params["p"] = copy_img(params["p"], params["cache"])
        print(params["p"])

    params, z = inference_main(params, models, augmenter, color_aug_fn)
    print(
        "::: finished or skipped inference after",
        timedelta(seconds=timer() - start_time),
    )
    process_timer = timer()

    # Stitch tiles together and postprocess to get instance segmentation
    print("Running post-processing")
    post_process_main(
        params,
        z,
    )
    print(f"Post-processing took: {timedelta(seconds=timer() - process_timer)}")
    print(f"Total inference time: {timedelta(seconds=timer() - start_time)}")

    # remove temporary files
    try:
        os.remove(params["model_out_p"] + "_inst.zip")
        os.remove(params["model_out_p"] + "_cls.zip")
        shutil.rmtree(params["output_dir"])

    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    print("Done")
    sys.exit(0)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="path to wsi, glob pattern or text file containing paths",
        required=True,
    )
    parser.add_argument(
        "--output", type=str, default=None, help="output directory", required=True
    )
    parser.add_argument(
        "--cp",
        type=str,
        default=None,
        help="comma-separated list of checkpoint folders to consider",
    )
    parser.add_argument(
        "--only_inference",
        action="store_true",
        help="split inference to gpu and cpu node/ only run inference",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--tta",
        type=int,
        default=4,
        help="test time augmentations, number of views (4= results from 4 different augmentations are averaged for each sample)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="tile size, models are trained on 256x256",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.96875,
        help="overlap between tiles, at 0.5mpp, 0.96875 is best, for 0.25mpp use 0.9375 for better results",
    )
    parser.add_argument(
        "--inf_workers",
        type=int,
        default=4,
        help="number of workers for inference dataloader, maximally set this to number of cores",
    )
    parser.add_argument(
        "--inf_writers",
        type=int,
        default=2,
        help="number of writers for inference dataloader, default 2 should be sufficient"
        + ", \ tune based on core availability and delay between final inference step and inference finalization",
    )
    parser.add_argument(
        "--pp_tiling",
        type=int,
        default=8,
        help="tiling factor for post processing, number of tiles per dimension, 8 = 64 tiles",
    )
    parser.add_argument(
        "--pp_overlap",
        type=int,
        default=256,
        help="overlap for postprocessing tiles, put to around tile_size",
    )
    parser.add_argument(
        "--pp_workers",
        type=int,
        default=16,
        help="number of workers for postprocessing, maximally set this to number of cores",
    )
    parser.add_argument("--cache", type=str, default=None, help="cache path")
    params = vars(parser.parse_args())
    main(params)
