import numpy as np
import random
import os
from functools import wraps
from tifffile import imread as tif_imread
from nd2 import imread as nd2_imread
import time
from tqdm import tqdm   

def crop_large_image(image, n_segments_x=1, n_segments_y=1):
    """
    Crops a large image (NumPy or Dask array) into smaller patches.
    Args:
        image (np.ndarray or dask.array): The input image to crop.
        n_segments_x (int): The number of segments to divide the image into (x-axis).
        n_segments_y (int): The number of segments to divide the image into (y-axis).
    Returns:
        list: A list of cropped image patches (same type as input).
    """
    height, width = image.shape[:2]
    segment_height = height // n_segments_y
    segment_width = width // n_segments_x
    cropped_images = []
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            start_row = i * segment_height
            start_col = j * segment_width
            end_row = start_row + segment_height
            end_col = start_col + segment_width
            cropped_image = image[start_row:end_row, start_col:end_col]
            cropped_images.append(cropped_image)
    return cropped_images

def stitch_images(images, n_segments_x=1, n_segments_y=1):
    """
    Stitches a list of cropped images back into a single large image.
    Args:
        images (list): A list of cropped image patches.
        n_segments_x (int): The number of segments the original image was divided into (x-axis).
        n_segments_y (int): The number of segments the original image was divided into (y-axis).
    Returns:
        np.ndarray: The stitched large image.       
    """
    segment_height = images[0].shape[0]
    segment_width = images[0].shape[1]
    stitched_image = np.zeros((segment_height * n_segments_y, segment_width * n_segments_x), dtype=images[0].dtype)
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            index = i * n_segments_x + j
            start_row = i * segment_height
            start_col = j * segment_width
            stitched_image[start_row:start_row + segment_height, start_col:start_col + segment_width] = images[index]
    return stitched_image

def convert_to_minimal_format(array):
    """
    Converts the input arrays to a minimal format by reducing their data type. If the input is a list,
    it will be converted to a NumPy array first.
    Args:
        array (np.ndarray/list of np.ndarray): The input arrays to convert.
    Returns:
        array (np.ndarray): Input array with minimal data type
    """
    # Convert masks to minimal data type
    if np.all(np.asarray(array)< 255):
        # If the masks are not in uint8 format, convert them to uint8
        array = np.asarray(array).astype(np.uint8)
    elif np.all(np.asarray(array)< 65535):
        # If the masks are not in uint16 format, convert them to uint16
        array = np.asarray(array).astype(np.uint16)
    elif np.all(np.asarray(array)< 4294967295):
        # If the masks are not in uint32 format, convert them to uint32
        array = np.asarray(array).astype(np.uint32)
    else:
        # If the masks are not in a standard format, convert them to a standard format
        array = np.asarray(array)
    return array

if __name__ == "__main__":
    # Test the utility functions
    from aicsimageio import AICSImage
    import numpy as np
    from tkinter.filedialog import askopenfilename
    from napari import Viewer
    from tkinter import Tk
    # Make root window and make it hidden
    root = Tk()
    root.withdraw()  # Hide the root window
    root.call('wm', 'attributes', '.', '-topmost', True) # Keep the window on top

    path = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.tif;*.nd2")])
    v = Viewer(title="Util tests", axis_labels=["Y", "X", "Tile"])
    # Test reading an AICS image
    image = AICSImage(path)
    print("Image shape:", image.shape)
    img_data = image.get_image_data("YXC", C=1).squeeze()  # Get a single channel image
    v.add_image(img_data, name="Original Image")

    # Test cropping function
    cropped = crop_large_image(img_data, n_segments_x=2, n_segments_y=2)
    print("Number of cropped images:", len(cropped))
    v.add_image(np.asarray(cropped), name="Cropped Images")

    # Test stitching function
    stitched = stitch_images(cropped, n_segments_x=2, n_segments_y=2)
    print("Stitched image shape:", stitched.shape)
    v.add_image(stitched, name="Stitched Image")

    # Test minimal format conversion
    minimal = convert_to_minimal_format(img_data)
    print("Minimal format shape:", minimal.shape)