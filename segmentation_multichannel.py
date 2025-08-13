from cellpose import models, denoise
from cellpose import version as cellpose_version
import os
#import napari
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog as sd
import tkinter.filedialog as fd
from pandas import DataFrame, read_csv, concat
import time
import nd2
import tifffile
from skimage.measure import regionprops_table
from skimage.exposure import rescale_intensity
from skimage.util import dtype_limits
from pandas import DataFrame
from tqdm import tqdm
from readlif.reader import LifFile
from aicsimageio import imread, AICSImage
#PROPERTIES = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std','bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')
PROPERTIES = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std')

def read_any_format_to_numpy(file_path):
    """
    Reads an image or video file and converts it to a NumPy array.

    Args:
        file_path (str): The path to the image or video file.

    Returns:
        list of np.ndarray: The image or video frames as a list of NumPy arrays.
        list of str: The names of the images if applicable (e.g., for .lif files).
    """
    if file_path.endswith('.avi'):
        return read_avi_to_grayscale_numpy(file_path)
    elif file_path.endswith('.nd2'):
        image = nd2.imread(file_path, xarray=True, dask=True)
        # If there are multiple channels, only take the Trans channel
        if 'C' in image.dims:
            image = image.sel({'C':'Trans'}).squeeze()
        if image.ndim == 2:
            # If the image is 2D, return it as a list with one element
            return [np.array(image)]
        elif 'P' in image.dims and image.ndim == 3:
            # If the image is 3D with a 'P' dimension, convert it to a list of 2D arrays
            return [np.array(image.sel({'P':p})) for p in image.coords['P']]
        elif 'T' in image.dims and image.ndim == 3:
            # If the image has a time dimension, convert it to a list of 2D arrays
            return [np.array(image.sel({'T':t})) for t in image.coords['T']]
        elif image.ndim == 4 and 'P' in image.dims and 'T' in image.dims:
            print("Unsupported ND2 file format with positions and timelapse not yet supported.")
            print("Processing only the first position and time frame.")
            return [np.array(image.isel(P=0, T=0))], None
    elif file_path.endswith('.lif'):
        new = LifFile(file_path)
        images = []
        names = []
        for image in new.get_iter_image():
            if "Merged" in image.name:  
                channels = [np.asarray(channel) for channel in image.get_iter_c()]
                images.append(channels)
                names.append(image.name)
        return images, names
        # do stuff
        # with LifFile(file_path) as lif:
        #     images = []
        #     for series in lif.series:
        #         for image in series.images:
        #             images.append(image.asarray())
        #     return images    
    elif file_path.endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(file_path) as tif:
            images = [page.asarray() for page in tif.pages]
        return images, None  # Return images as a list and None for names
    elif file_path.endswith('.npy'):
        return list(np.load(file_path)), None  # Load the .npy file and return as a list
    else:
        raise ValueError("Unsupported file format.")

def read_avi_to_grayscale_numpy(video_path):

    """
    Reads an AVI video file, converts its frames to grayscale,
    and returns a list of these grayscale frames as NumPy ndarrays.

    Args:
        video_path (str): The path to the .avi video file.

    Returns:
        list: A list of NumPy ndarrays, where each ndarray represents a
              grayscale frame. Returns an empty list if the video
              cannot be opened or read.
    """
    frames_list = []

    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    # 2. Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return frames_list

    # 3. Read frames in a loop
    while True:
        ret, frame = cap.read()

        if not ret:
            # End of video or error reading frame
            break

        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # The 'grayscale_frame' is a NumPy ndarray
        frames_list.append(grayscale_frame)


    # 4. Release the VideoCapture object
    cap.release()


    return frames_list
def convert_multichannel_to_grayscale(image_list):
    """
    Converts a list of multichannel images to grayscale by averaging the channels.

    Args:
        image_list (list): A list of multichannel images as lists of NumPy ndarrays.

    Returns:
        list: A list of grayscale images as NumPy ndarrays.
    """
    grayscale_images = []
    for image in image_list:
        if isinstance(image, list):
            # If the image is a list of arrays, average them
            grayscale_image = np.mean(image, axis=0)
        if isinstance(grayscale_image, np.ndarray):
            # If the image is ndarray append it
            grayscale_images.append(grayscale_image)
        else:
            raise ValueError("Image must be a NumPy ndarray.")
    return grayscale_images
def convert_multichannel_to_maxproj(image_list):
    """
    Converts a list of multichannel images to composite by averaging the channels.

    Args:
        image_list (list): A list of multichannel images as lists of NumPy ndarrays.

    Returns:
        list: A list of maxprojection images as NumPy ndarrays.
    """
    maxproj_images = []
    for image in image_list:
        if isinstance(image, list):
            # If the image is a list of arrays, take the maximum projection
            maxproj_image = np.max(image, axis=0)
        if isinstance(maxproj_image, np.ndarray):
            # If the image is ndarray append it
            maxproj_images.append(maxproj_image)
        else:
            raise ValueError("Image must be a NumPy ndarray.")
    return maxproj_images
           
if __name__ == "__main__":

    # Make root window and make it hidden
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.call('wm', 'attributes', '.', '-topmost', True) # Keep the window on top
    #Get user inputs
    files = fd.askopenfilenames(title='Select files')
    if not files:
        messagebox.showerror("Error", "No files selected. Exiting.")
        root.destroy()
        exit()
    savedir = fd.askdirectory(title='Select the directory to save the masks')
    if not savedir:
        savedir = os.path.dirname(files[0])  # Use the directory of the first file if no directory is selected
    # Ask for the model diameter
    model_diameter = sd.askinteger("Diameter", "Enter the diameter of the cells (in pixels):", initialvalue=30)
    # Ask whether to get regionprops
    get_regionprops = messagebox.askyesno("Regionprops", "Get region properties for the masks?")

    # Custom or default model?
    model_choice = messagebox.askyesno("Custom Model", "Use a custom model?")
    if model_choice:
        # If the user chooses to use a custom model, ask for the model file
        model_path = fd.askopenfilename(title="Select a Cellpose model") 
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    else:
        if cellpose_version >= '4.0':
            # Use cellposeSAM model
            model = models.CellposeModel(gpu=True)
        elif cellpose_version < '4.0':
            # Use cyto3 model on its own
            model = models.CellposeModel(gpu=True, pretrained_model='cyto3')
    root.destroy()  # Destroy the root window after getting inputs
    # endregion
    # region: Main processing loop
    # Loop through the selected files and process each one
    all_props = []  # Initialize the list to store regionprops DataFrames
    start_all = time.perf_counter()  # Start the timer for all files
    for i, file_path in enumerate(tqdm(files, desc="Processing files", unit="file")):
        start_file = time.perf_counter()  # Start the timer
        # Read images. segment and save masks
        try:
            image_list, names = read_any_format_to_numpy(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        if names is not None:
            images_composite = convert_multichannel_to_maxproj(image_list)
        # If image multichannel convert to composite
        if isinstance(image_list, list):
            if isinstance(image_list[0], np.ndarray):
                if len(image_list) > 1:
                    # Average all the channels
                    images_composite = [np.mean(image_list, axis=0)]
                elif len(image_list) == 1:
                    images_composite = [image_list[0]]
            elif isinstance(image_list[0], list):
                # If the image is a list of lists, convert each sublist to a numpy array
                images_composite = [np.mean(img,axis=0) for img in image_list]
        elif isinstance(image_list, np.ndarray):
            images_composite = [image_list]

        print(f"\nProcessing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        for i, image_composite in enumerate(images_composite):
            
            start_segmentation = time.perf_counter()  # Start the timer
            ## Perform segmentation in parallel on all images in the list
            masks, flows, styles = model.eval(image_composite, diameter=model_diameter)
            if max(masks)<255:
                masks = np.array(masks).astype(np.uint8)
            end_segmentation = time.perf_counter()  # End the timer
            print(f"Segmentation time for {os.path.basename(file_path)} : {end_segmentation - start_segmentation:.2f} seconds")    
            
            # Generate the file name without the extension 
            file = os.path.basename(file_path).split(".")[0] + names[i] if names else os.path.basename(file_path).split(".")[0]
            # Make image uint8 for saving
            image = np.array(image_list[i]).squeeze()
            # Scale the image to uint8 if it is not already using skimage
            # if image.dtype != np.uint8:
            #     try:
            #         image = rescale_intensity(image, in_range='dtype', out_range=(0, 255)).astype(np.uint8)
            #     except Exception as e:
            #         print(f"Error rescaling image: {e}")

            if get_regionprops:
                print("Getting regionprops for the masks...")
                # Get regionprops for the masks
                image = np.moveaxis(np.asarray(image_list[i]).squeeze(), 0, -1)  # Move the channel axis to the last position
                props = regionprops_table(masks, image, properties=PROPERTIES)
                df = DataFrame(props)
                df['file'] = file  # or 'z' for z-stack
                all_props.append(df)
                    

            # Save the masks to a file or process them as needed
            np.save(os.path.join(savedir, f"masks_d-{model_diameter}_{file}.npy"), masks)
            filename = f"masks_d-{model_diameter}_{file}.npy"
            print(f"Masks and image.npy save path: {os.path.join(savedir, filename)}")
            # Save the masks to a tiff file
            tifffile.imwrite(os.path.join(savedir, f"masks_d-{model_diameter}_{file}.tif"), masks.astype(np.uint8))
            # Move the channel axis to the first position if it exists
            if image.ndim == 3:
                image = np.moveaxis(image, -1, 0)
            np.save(os.path.join(savedir, f"image_{file}.npy"),image)
            endfile = time.perf_counter()  # End the timer
            print(f"Total time for {os.path.basename(file_path)}: {endfile - start_file:.2f} seconds")
    props_df = concat(all_props, ignore_index=True)
    # Save the regionprops to a CSV file
    props_df.to_csv(os.path.join(savedir, f"regionprops.csv"), index=False)
    print(f"Regionprops saved to {os.path.join(savedir, f'regionprops.csv')}")
    endall = time.perf_counter()  # End the timer for all files
    print(f"Total time for all files: {endall - start_all:.2f} seconds")
    # endregion