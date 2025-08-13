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
from skimage.measure import regionprops_table, label
from skimage.exposure import rescale_intensity
from skimage.util import dtype_limits
from skimage.filters import gaussian, median
from pandas import DataFrame
from tqdm import tqdm
from readlif.reader import LifFile  
import xarray as xr
from aicsimageio import AICSImage
from utils import crop_large_image, stitch_images, convert_to_minimal_format
#from skimage.restoration import gaussian_denoise, estimate_sigma
PROPERTIES_ALL = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std','bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')
PROPERTIES_MINIMAL = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std')
PROPERTIES_NO_INTENSITY = ('label', 'area', 'perimeter', 'centroid', 'bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')   

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def read_lif_to_xarray_with_metadata(file_path):
    """
    Reads a Leica .lif file and returns an xarray.DataArray with dimension names
    inferred from the metadata (not assumed).
    """
    with LifFile(file_path) as lif:
        positions = []
        dims_list = []
        for series in lif.series:
            t_images = []
            for image in series.images:
                arr = image.asarray()
                # Try to get dimension names from image metadata
                # readlif.reader does not provide explicit names, but you can infer:
                # If arr.ndim == 4: (C, Z, Y, X) or (C, T, Y, X)
                # If arr.ndim == 3: (C, Y, X)
                # If arr.ndim == 2: (Y, X)
                # You may need to check image.dims or image.shape if available
                if hasattr(image, 'dims'):
                    dims = image.dims  # e.g. ('C', 'Z', 'Y', 'X')
                else:
                    # Fallback: guess based on shape
                    if arr.ndim == 4:
                        dims = ('C', 'Z', 'Y', 'X')
                    elif arr.ndim == 3:
                        dims = ('C', 'Y', 'X')
                    elif arr.ndim == 2:
                        dims = ('Y', 'X')
                    else:
                        raise ValueError("Unknown dimension order for LIF image.")
                dims_list.append(dims)
                t_images.append(arr)
            if t_images:
                t_stack = np.stack(t_images, axis=0)  # stack along new T axis
                positions.append(t_stack)
        if not positions:
            raise ValueError("No images found in lif file.")
        data = np.stack(positions, axis=0)  # stack positions
        # Compose dimension names: add 'P' (position) and 'T' (time) if needed
        # Use the first dims as reference
        base_dims = dims_list[0]
        dims = ('P', 'T') + base_dims if len(data.shape) == len(base_dims) + 2 else ('P',) + base_dims
        coords = {d: np.arange(s) for d, s in zip(dims, data.shape)}
        da = xr.DataArray(data, dims=dims, coords=coords)
        return da

def read_tiff_to_xarray_with_metadata(file_path):
    """
    Reads a TIFF file and returns an xarray.DataArray with dimension names
    inferred from the metadata (not assumed).
    """
    with tifffile.TiffFile(file_path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        # Get axes string from metadata, e.g. 'TCYX', 'CZYX', etc.
        axes = getattr(series, 'axes', None)
        if axes is None:
            # Fallback: guess based on ndim
            if arr.ndim == 5:
                axes = 'PTCYX'
            elif arr.ndim == 4:
                axes = 'TCYX'
            elif arr.ndim == 3:
                axes = 'CYX'
            elif arr.ndim == 2:
                axes = 'YX'
            else:
                raise ValueError("Unknown TIFF axes order.")
        dims = tuple(axes)
        coords = {d: np.arange(s) for d, s in zip(dims, arr.shape)}
        da = xr.DataArray(arr, dims=dims, coords=coords)
        return da

def read_any_format_to_numpy_multidim(file_path):
    """
    Reads images or timelapses from a file and converts them to a lists of NumPy arrays that 
    can be processed by Cellpose.

    Args:
        file_path (str): The path to the image or video file.

    Returns:
        list of lists of np.ndarray: The timelapse as a list of NumPy arrays, 
        nested in a list to allow for multiple positions.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.avi'):
        return read_avi_to_grayscale_numpy(file_path)
    elif file_path.endswith('.nd2'):
        image = nd2.imread(file_path, xarray=True, dask=True)
        # If there are multiple channels, only take the Trans channel
        if image.ndim == 2:
            # If the image is 2D, return it as a list with one element
            return [[np.array(image)]]
        elif image.ndim == 3:
            if 'C' in image.dims:
                return [[np.array(image)]]
            elif 'P' in image.dims:
                # If the image is 3D with a 'P' dimension, convert it to a list of 2D arrays
                return [[np.array(image.sel({'P':p})) for p in image.coords['P']]]
            elif 'T' in image.dims:
                # If the image has a time dimension, convert it to a list of 2D arrays
                return [[np.array(image.sel({'T':t})) for t in image.coords['T']]]
        elif image.ndim == 4:
            if 'P' in image.dims and 'T' in image.dims:
                # go through all positions and time frames
                #return [[np.array(image.sel({'P':p, 'T':t})) for t in image.coords['T']] for p in image.coords['P']]
                image_sets = []
                for p in image.coords['P']:
                    # Load all time points for this position at once (still chunked)
                    arr = np.array(image.sel({'P': p}))
                    # arr shape: (T, Y, X)
                    time_slices = [arr[t] for t in range(arr.shape[0])]
                    image_sets.append(time_slices)
                return image_sets
    elif file_path.endswith('.lif'):
        with LifFile(file_path) as lif:
            images = []
            for series in lif.series:
                for image in series.images:
                    images.append(image.asarray())
            return [images]
    elif file_path.endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(file_path) as tif:
            images = [page.asarray() for page in tif.pages]
        return [images]
    elif file_path.endswith('.npy'):
        return [list(np.load(file_path))]  # Load the .npy file and return as a list
    else:
        raise ValueError("Unsupported file format.")

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
    # elif file_path.endswith('.lif'):
    #     with LifFile(file_path) as lif:
    #         images = []
    #         for series in lif.series:
    #             for image in series.images:
    #                 images.append(image.asarray())
    #         return images    
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



def process_files(files, model, model_diameter, savedir, get_regionprops=False):
    """
    Processes a list of files, reads images, segments them using the Cellpose model,
    and saves the masks and region properties to the specified directory.
    Args:
        files (list): List of file paths to process.
        model (CellposeModel): The Cellpose model to use for segmentation.
        model_diameter (int): The diameter of the cells for the model.
        savedir (str): Directory where the masks and region properties will be saved.
        get_regionprops (bool): Whether to compute and save region properties.
    """
    # Process the selected files
    # region
    # Loop through the selected files and process each one
    start_all = time.perf_counter()  # Start the timer for all files
    for i, file_path in enumerate(tqdm(files, desc="Processing files", unit="file")):
        # Read images. segment and save masks
        image_sets = read_any_format_to_numpy_multidim(file_path)
        for j, image_list in enumerate(tqdm(image_sets, desc=f"Processing image sets", unit="image")):
            ## Perform segmentation in parallel on all images in the list
            masks, flows, styles = model.eval(image_list, diameter=model_diameter)
            #image=np.array(image_list).astype(np.uint8).squeeze()
            if np.all(np.array(masks)< 255):
                # If the masks are not in uint8 format, convert them to uint8
                masks = np.array(masks).astype(np.uint8)
            else:
                masks = np.array(masks)
            # Generate the file name without the extension
            file = os.path.basename(file_path).split('.')[0] + f"_{j+1}"  # Append the image set index to the file name
            # Scale the image to uint8 if it is not already using skimage
            image = np.array(image_list).squeeze()
            if image.dtype != np.uint8:
                image = rescale_intensity(image, in_range='dtype', out_range=(0, 255)).astype(np.uint8)
     
            if get_regionprops:
                print("Getting regionprops for the masks...")
                # Get regionprops for the masks 
                all_props = []
                for i, mask in enumerate(masks):
                    props = regionprops_table(mask, intensity_image=image,properties=PROPERTIES_MINIMAL)
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
    endall = time.perf_counter()  # End the timer for all files
    print(f"Total time for all files: {endall - start_all:.2f} seconds")
    # endregion

@timeit
def process_files_multichannel(files, model, model_diameter, savedir, savefix, get_regionprops=False, channels=[2, 0]):
    """
    Processes a list of files with multiple channels, segments them using the Cellpose model,
    and saves the masks and region properties to the specified directory.
    Args:
        files (list): List of file paths to process.
        model (CellposeModel): The Cellpose model to use for segmentation.
        model_diameter (int): The diameter of the cells for the model.
        savedir (str): Directory where the masks and region properties will be saved.
        savefix (str): Suffix to append to the saved file names.
        get_regionprops (bool): Whether to compute and save region properties.
        channels (list): List of channel indices to process. First index is the channel to segment, second onward index is the channel to use for regionprops intensity metrics.
    """
    # Process the selected files
    # region
    # Loop through the selected files, read imagges and structure them for Cellpose       
    for i, file_path in enumerate(tqdm(files, desc="Reading files", unit="file")):
        # Read images. segment and save masks
        image_sets= read_any_format_to_numpy_multidim(file_path)
        if i == 0:
            # Check number of channels
            num_channels = len(image_sets[0])  # Assuming all image sets have the same number of channels
            channel_sets = [[] for _ in range(num_channels)]
        for j in range(num_channels):
            channel_sets[j].append(image_sets[0][j])  # Append each channel to the respective list

    
    ## Perform segmentation in parallel on all images in the list
    masks, _, _ = model.eval(channel_sets[channels[0]], diameter=model_diameter, cellprob_threshold=-0.1, niter=400)
    
    # Convert masks to minimal data type
    if np.all(np.array(masks)< 255):
        # If the masks are not in uint8 format, convert them to uint8
        masks = np.array(masks).astype(np.uint8)
    elif np.all(np.array(masks)< 65535):
        # If the masks are not in uint16 format, convert them to uint16
        masks = np.array(masks).astype(np.uint16)
    elif np.all(np.array(masks)< 4294967295):
        # If the masks are not in uint32 format, convert them to uint32
        masks = np.array(masks).astype(np.uint32)
    else:
        # If the masks are not in a standard format, convert them to a standard format
        masks = np.array(masks)
    
    # # Scale the image to uint8 if it is not already using skimage
    # image = np.array(image_list).squeeze()
    # if image.dtype != np.uint8:
    #     image = rescale_intensity(image, in_range='dtype', out_range=(0, 255)).astype(np.uint8)

    if get_regionprops:
        # Get regionprops for the masks
        all_props = []
        try:
            for i, mask in enumerate(tqdm(masks,desc="Calculating regionprops", unit="mask")):
                props = regionprops_table(mask, intensity_image=channel_sets[channels[1]][i],properties=PROPERTIES_MINIMAL)
                df = DataFrame(props)
                df['file'] = os.path.basename(files[i])  
                all_props.append(df)
            props_df = concat(all_props, ignore_index=True)
        except Exception as e:
            print(f"Error occurred while calculating regionprops: {e}")
            props_df = None

    # Saving data part
    # Generate the file name without the extension
    file = savefix if savefix else os.path.basename(files[0]).split('.')[0]  # Use the first file name as a base
    # Save the masks to a file or process them as needed
    filename = f"masks_d-{model_diameter}_{file}.npy"
    np.save(os.path.join(savedir, filename), masks)
    print(f"Masks and image.npy save path: {os.path.join(savedir, filename)}")
    # Save the images as a numpy array stack
    image = np.array(channel_sets[channels[0]]).squeeze()
    intensity = np.array(channel_sets[channels[1]]).squeeze()
    np.save(os.path.join(savedir, f"image_{file}.npy"), image)
    np.save(os.path.join(savedir, f"intensity_{file}.npy"), intensity)

    try:
        if props_df is not None:
            # Save the regionprops to a CSV file
            props_df.to_csv(os.path.join(savedir, f"regionprops_{file}.csv"), index=False)
            print(f"Regionprops saved to {os.path.join(savedir, f'regionprops_{file}.csv')}")
    except Exception as e:
        print(f"Error occurred while saving regionprops: {e}")

        # endregion

@timeit
def process_files_multichannel_2(
    files, model, model_diameter, savedir, savefix,
    get_regionprops=True, channels=[0, 1], subset=slice(None),
    chunksize=[1, 1]
):
    """
    Processes a list of files with multiple channels using AICSImage, segments them using the Cellpose model,
    and saves the masks and region properties to the specified directory.
    Args:
        files (list): List of file paths to process.
        model (CellposeModel): The Cellpose model to use for segmentation.
        model_diameter (int): The diameter of the cells for the model.
        savedir (str): Directory where the masks and region properties will be saved.
        savefix (str): Suffix to append to the saved file names.
        get_regionprops (bool): Whether to compute and save region properties.
        channels (list): List of channel indices to process. First index is the channel to segment, second onward index is/are the channel(s) to use for regionprops intensity metrics.
        subset (slice): Slice object to select a subset of images from the cropped image list. Is slice(None) by default, which means all images will be processed.
        chunksize (list): List of chunk sizes for processing the images. The first element is the chunk size for the x dimension, the second element is the chunk size for the y dimension.
    """
    # from pandas import DataFrame, concat
    # import numpy as np
    # import os
    # from skimage.measure import regionprops_table
    # from tqdm import tqdm

    for file_path in tqdm(files, desc="Processing files", unit="file"):
        try:
            img = AICSImage(file_path)
        except:
            if file_path.endswith(".npy"):
                data=np.load(file_path)
            img = AICSImage(data)
        for scene in img.scenes:
            img.set_scene(scene)
            # Read segmentation channel (as 2D or 3D if needed)
            seg_channel = channels[0]
            seg_img = img.get_image_dask_data("YX", C=seg_channel)
            # Apply median filter to denoise the segmentation channel
            #seg_img = median(seg_img)
            seg_list = crop_large_image(seg_img, n_segments_x = chunksize[0], n_segments_y = chunksize[1])  # Crop the image into smaller patches if needed
            seg_list = seg_list[subset]
            seg_list = [patch.compute() for patch in seg_list]

            #seg_img_np = np.array(seg_img).squeeze()  # shape: (Y, X)

            # Segment
            masks, _, _ = model.eval(seg_list, diameter=model_diameter, min_size=0)
            
            
            # Save masks and images
            base = os.path.splitext(os.path.basename(file_path))[0]
            if len(base) > 50:
                base = base[:50]  # Truncate base name to 50 characters if too long
            if savefix:
                base = base + f"_{savefix}"
            filename = f"masks_d-{model_diameter}_{base}.npy"

            # Save masks
            if subset == slice(None):
                # If no subset is specified, stitch all masks together
                masks = stitch_images(masks, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                # Relabel masks
                masks = label(masks)
                masks = convert_to_minimal_format(masks)  # Convert to minimal format
                np.save(os.path.join(savedir, filename), masks)
                np.save(os.path.join(savedir, f"image_{base}.npy"), seg_img)
                print(f"Masks and image.npy save path: {os.path.join(savedir, filename)}")
            else:
                # If a subset is specified, save the masks as stack
                masks = convert_to_minimal_format(masks)  # Convert to minimal format
                np.save(os.path.join(savedir, filename), masks)
                np.save(os.path.join(savedir, f"image_{base}.npy"), seg_list)
                print(f"Masks and image.npy save path: {os.path.join(savedir, filename)}")

            # Get intensity data and save as npy
            # Read all regionprops channels and stack as last axis
            intensity_img = img.get_image_dask_data("YXC", C=channels[1:])  # shape: (Y, X, C)
            intensity_list = crop_large_image(intensity_img, n_segments_x = chunksize[0], n_segments_y = chunksize[1])  # Crop the image into smaller patches if needed
            intensity_list = intensity_list[subset]
            # Regionprops
            props_df = None
            if get_regionprops:

                all_props = []
                try:
                    # If masks is 3D (batch), loop; else, just one mask
                    if masks.ndim == 3:
                        mask_list = masks
                    else:
                        mask_list = [masks]
                        intensity_list = [intensity_img]
                    for i, (mask, intensity_image) in enumerate(zip(mask_list, intensity_list)):
                        props = regionprops_table(
                            mask,
                            intensity_image=intensity_image.compute(),  # Ensure intensity image is computed
                            properties=PROPERTIES_MINIMAL
                        )
                        df = DataFrame(props)
                        df['file'] = os.path.basename(file_path)
                        df['scene'] = scene
                        df['frame'] = i
                        all_props.append(df)
                    props_df = concat(all_props, ignore_index=True)
                except Exception as e:
                    print(f"Error occurred while calculating regionprops: {e}")
                    props_df = None
            # Save regionprops
            if props_df is not None:
                try:
                    props_df.to_csv(os.path.join(savedir, f"regionprops_{base}.csv"), index=False)
                    print(f"Regionprops saved to {os.path.join(savedir, f'regionprops_{base}.csv')}")
                except Exception as e:
                    print(f"Error occurred while saving regionprops: {e}")
        
        if subset == slice(None):
            # Move channel axis to the beginning
            intensity_img = np.moveaxis(intensity_img, -1, 0)
            np.save(os.path.join(savedir, f"intensity_{base}.npy"), intensity_img)
        else:
            # Move channel axis to the beginning for each image in the list
            intensity_list = [np.moveaxis(img, -1, 0).squeeze() for img in intensity_list]
            np.save(os.path.join(savedir, f"intensity_{base}.npy"), intensity_list)

@timeit
def restore(
    image
):
    """ Applies noise removal and background extraction on image
    Args:
        image (np.ndarray): image to be processed
    Returns:
        restored (np.ndarray): restored image
    """

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
    savefix = sd.askstring("Savefix", "Enter a suffix for the saved files (optional):", initialvalue=files[0].split('/')[-2])
    # Check cellpose version
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
    # Process the files
    process_files_multichannel_2(files, model, model_diameter, savedir, savefix=savefix, 
                                 get_regionprops=get_regionprops, channels=[0, 1], 
                                 subset=slice(None), chunksize=[10,10])  
    # endregion
    #
