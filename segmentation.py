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

def read_any_format_to_numpy(file_path):
    """
    Reads an image or video file and converts it to a NumPy array.

    Args:
        file_path (str): The path to the image or video file.

    Returns:
        list of np.ndarray: The image or video frames as a list of NumPy arrays.
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
            return [np.array(image.isel(P=0, T=0))]
        
    elif file_path.endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(file_path) as tif:
            images = [page.asarray() for page in tif.pages]
        return images
    elif file_path.endswith('.npy'):
        return list(np.load(file_path))  # Load the .npy file and return as a list
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
    # Check cellpose version
    if cellpose_version < '4.0':
        denoised = messagebox.askyesno("Denoise", "Denoise the images?")
    else:
        denoised = False
    # Ask if the user wants to segment the images
    segmented = messagebox.askyesno("Segment", "Segment the images?")
    # Ask for the model diameter
    model_diameter = sd.askinteger("Diameter", "Enter the diameter of the cells (in pixels):", initialvalue=30)
    # Ask whether to get regionprops
    get_regionprops = messagebox.askyesno("Regionprops", "Get region properties for the masks?")
    if denoised and not segmented:
        # If the user chooses to denoise, create a denoise model
        model_denoise = denoise.DenoiseModel(gpu=True, model_type="denoise_cyto3")
    ## Define cellpose model for segmentation
    if segmented:
        # Custom or default model?
        model_choice = messagebox.askyesno("Custom Model", "Use a custom model?")
        if model_choice:
            # If the user chooses to use a custom model, ask for the model file
            model_path = fd.askopenfilename(title="Select a Cellpose model")
            if denoised:
                # Denoising only works with cellpose 3.1.1 currently
                model_denoise= denoise.CellposeDenoiseModel(gpu=True, pretrained_model= model_path,restore_type="denoise_cyto3")
            else:
                model = models.CellposeModel(gpu=True, pretrained_model=model_path)

        else:
            if denoised:
                # # Denoising only works with cellpose 3.1.1 currently so no cellposeSAM available
                model_denoise = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3", chan2_restore=False)
            else:    
                if cellpose_version >= '4.0':
                    # Use cellposeSAM model
                    model = models.CellposeModel(gpu=True)
                elif cellpose_version < '4.0':
                    # Use cyto3 model on its own
                    model = models.CellposeModel(gpu=True, pretrained_model='cyto3')
        
        
        # filter_choice = messagebox.askyesno("Filter files", "Filter the files based on a CSV file?")
        # if filter_choice:
        #     sort_path = fd.askopenfilename(title='Select the csv with the files to include', filetypes=[('CSV files', '*.csv')])
        #     if sort_path:
        #         df = read_csv(sort_path)
        #         df = df.where(df['Free']=='Yes').dropna(how='any')
        #         list_of_files = df['File'].tolist()
        #         files = [f for f in files if os.path.basename(f).split(' ')[1] in list_of_files]

    root.destroy()  # Destroy the root window after getting inputs
    # endregion
    # 
    # Process the selected files
    # region
    # Loop through the selected files and process each one
    start_all = time.perf_counter()  # Start the timer for all files
    for i, file_path in enumerate(tqdm(files, desc="Processing files", unit="file")):
        start_file = time.perf_counter()  # Start the timer
        # Read images. segment and save masks
        image_list = read_any_format_to_numpy(file_path)
        print(f"\nProcessing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        
        if segmented:
            start_segmentation = time.perf_counter()  # Start the timer
            ## Perform segmentation in parallel on all images in the list
            if denoised:
                masks, flows, styles, image_list = model_denoise.eval(image_list, channels=[0,0], diameter=model_diameter)
                #imgs_dn = np.asarray(imgs_dn).squeeze()
                #v.add_image(np.asanyarray(imgs_dn), name=os.path.basename(file_path) + "_denoised", blending='additive')
                #np.save(os.path.join(savedir, f"denoised_{os.path.basename(file_path)}.npy"), imgs_dn)

            else:
                masks, flows, styles = model.eval(image_list, diameter=model_diameter)

            #image=np.array(image_list).astype(np.uint8).squeeze()
            masks = np.array(masks).astype(np.uint8)
            end_segmentation = time.perf_counter()  # End the timer
            print(f"Segmentation time for {os.path.basename(file_path)}: {end_segmentation - start_segmentation:.2f} seconds")    
        elif denoised:
            start_denoising = time.perf_counter()
            # Only denoise the images
            image_list = model_denoise.eval(image_list, channels=[0,0])
            #v.add_image(np.asanyarray(image_list_den), name=os.path.basename(file_path) + "_denoised", blending='additive')
            # Convert the denoised images to a numpy array
            #image = np.asarray(image_list).squeeze()
            end_denoising = time.perf_counter()
            print(f"Denoising time for {os.path.basename(file_path)}: {end_denoising - start_denoising:.2f} seconds")
        # Generate the file name without the extension 
        try:
            file = os.path.basename(file_path).split(" ")[1]
        except:
            file = os.path.basename(file_path).split(".")[0]

        # Make image uint8 for saving
        image = np.array(image_list).squeeze()
        # Scale the image to uint8 if it is not already using skimage
        if image.dtype != np.uint8:
            image = rescale_intensity(image, in_range='dtype', out_range=(0, 255)).astype(np.uint8)
        if segmented:
            if get_regionprops:
                print("Getting regionprops for the masks...")
                # Get regionprops for the masks 
                all_props = []
                for i, mask in enumerate(masks):
                    props = regionprops_table(mask, properties=('label', 'area', 'perimeter', 'centroid', 'bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length'))
                    df = DataFrame(props)
                    df['frame'] = i  # or 'z' for z-stack
                    all_props.append(df)

            props_df = concat(all_props, ignore_index=True)
            # Save the regionprops to a CSV file
            props_df.to_csv(os.path.join(savedir, f"regionprops_{file}.csv"), index=False)
            print(f"Regionprops saved to {os.path.join(savedir, f'regionprops_{file}.csv')}")
            # Save the masks to a file or process them as needed
            np.save(os.path.join(savedir, f"masks_d-{model_diameter}_{file}.npy"), masks)
            filename = f"masks_d-{model_diameter}_{file}.npy"
            print(f"Masks and image.npy save path: {os.path.join(savedir, filename)}")
        
        np.save(os.path.join(savedir, f"image_{file}.npy"),image)
        endfile = time.perf_counter()  # End the timer
        print(f"Total time for {os.path.basename(file_path)}: {endfile - start_file:.2f} seconds")
    endall = time.perf_counter()  # End the timer for all files
    print(f"Total time for all files: {endall - start_all:.2f} seconds")
    # endregion