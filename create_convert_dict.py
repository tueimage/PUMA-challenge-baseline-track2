import os

def convert_dict(images_path, output):
    # List all TIFF images that don't end with '_context.tif'
    images = [f for f in os.listdir(images_path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    
    # Open the output file once, in append mode, outside the loop
    with open(output, 'a') as file:
        for count, image in enumerate(images, 1):  # Use enumerate for automatic counting
            img_name = f'Mark_{count:03}_0000.png'
            gt_name = f'Mark_{count:03}.png'
            file.write(f'{image}, {img_name}, {gt_name}\n')  # Properly format and write image names

# Example usage
images_path = '/input/images/melanoma-wsi'
output = 'convert_dict.txt'
convert_dict(images_path, output)
