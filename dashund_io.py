from tkinter import filedialog, messagebox
import nd2, os, tifffile
import numpy as np
from readlif.reader import LifFile
import xarray as xr
import dask.array as da
from aicsimageio import AICSImage, imread

# Class that extends AICSImage to add functions to output data compatible with cellpose
class dash_bioImage(AICSImage):
    def __init__(self, image, segment_channels=None, intensity_channels=None, reader = None, reconstruct_mosaic = True, fs_kwargs = ..., **kwargs):
        super().__init__(image, reader, reconstruct_mosaic, fs_kwargs, **kwargs)  
        
    def to_cellpose_format(self, segmentation_channels):
        """
        Convert the AICSImage instance to a format compatible with Cellpose.
        """
        # Get ALL channel data at once
        full_img = self.get_image_data("TCYX")  # Single read
        
        # Use views/slices instead of copying
        seg_data = full_img[:, segmentation_channels]  # View, not copy
                
        # Convert to list for Cellpose
        seg_list = [seg_data[t] for t in range(seg_data.shape[0])]  # Views of time points

        return seg_list
    def get_regionprops_input(self, intensity_channels = None, masks = None):
        """
        Convert the AICSImage instance to a format compatible with regionprops.

        Arguments:
            intensity_channels (list/touple of channel indexes): list of channels needed for intensity data
            masks (Optional ndarray/list of ndarrays): masks that will be used for regionprops. If provided,
                compatibility of masks with intensity will be checked.
        """
        if intensity_channels is None:
            raise ValueError("No channels were provided to extract intensity data")
        
        # Get ALL channel data at once
        full_img = self.get_image_data("TCYX")  # Single read
        
        intensity_data = full_img[:, intensity_channels] if intensity_channels else None
        
        # Convert to list for regionprops
        intensity_list = [intensity_data[t] for t in range(intensity_data.shape[0])] if intensity_data is not None else None

        if intensity_list is None:
            raise ValueError("There is no intensity data available.")
        elif masks is not None:
            #Check compatibility
            intensity_list_fixed = intensity_list
            masks_fixed = masks
            return intensity_list_fixed, masks_fixed
        
        return intensity_list
    def get_trackastra_format(self, masks):
        """

        """


def read_lif_to_xarray_with_metadata(file_path, dims=None, channel_names=None, dask=False):
    """
    Reads a Leica .lif file and returns a a list of xarray.DataArray.
    
    """
    lif = LifFile(file_path)
    arrays = []
    pos_names = []
    for image in lif.get_iter_image():
        pos_name = image.name
        if 'm' in image.dims._fields and image.dims._asdict()['m'] > 1:
            print(f"Skipping image '{pos_name}' with 'm' dimension (unmerged mosaic image).")
            continue  # Skip images with 'm' dimension (unmerged mosaic images)
        else:
            arr = np.stack([np.asarray(channel) for channel in image.get_iter_c()], axis=0)  # (C, Y, X)
            if dims is not None and len(dims) == arr.ndim:
                curr_dims = tuple(dims)
            elif hasattr(image, 'dims'):
                # Remove dimensions that are size 1 and add channel as the first dimension
                curr_dims = tuple(d for d in image.dims._fields if image.dims._asdict()[d] > 1)  # Tuple of dimension names, e.g. ('C', 'Y', 'X')
                if 'c' not in curr_dims:
                    curr_dims = ('c',) + curr_dims
            else:
                if arr.ndim == 4:
                    curr_dims = ('c','z', 'y', 'x')
                elif arr.ndim == 3:
                    curr_dims = ('c', 'y', 'x')
                elif arr.ndim == 2:
                    curr_dims = ('y', 'x')
                else:
                    raise ValueError("Unknown dimension order for LIF image.")
            if channel_names is not None and 'c' in curr_dims:
                coords = {d: channel_names if d == 'c' else np.arange(s) for d, s in zip(curr_dims, arr.shape)}
            else:
                coords = {d: np.arange(s) for d, s in zip(curr_dims, arr.shape)}
            if dask:
                arr_xr = xr.DataArray(da.from_array(arr, chunks='auto'), dims=curr_dims, coords=coords)
            else:
                arr_xr = xr.DataArray(arr, dims=curr_dims, coords=coords)
            # Add scale as metadata if available
            if hasattr(image, 'scale_n'):
                arr_xr.attrs['pixel_size_um'] = {image.dims._fields[key-1]: value for key, value in image.scale_n.items()}
            arr_xr.attrs['name'] = pos_name
            arrays.append(arr_xr)
            pos_names.append(pos_name)
    if not arrays:
        raise ValueError("No images found in lif file.")
    return arrays

def read_tiff_to_xarray_with_metadata(file_path, dims=None, dask=False):
    """
    Reads a TIFF file and returns an xarray.DataArray.
    If there are multiple positions, stacks them along a new 'P' dimension.
    """
    with tifffile.TiffFile(file_path) as tif:
        arrays = []
        pos_names = []
        for idx, series in enumerate(tif.series):
            arr = da.from_zarr(series.aszarr()) if dask else series.asarray()
            axes = getattr(series, 'axes', None)
            if axes is None:
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
            dims_out = tuple(axes)
            coords = {d: np.arange(s) for d, s in zip(dims_out, arr.shape)}
            arr_xr = xr.DataArray(arr, dims=dims_out, coords=coords)
            arrays.append(arr_xr)
            pos_name = getattr(series, 'name', f'Position_{idx}')
            pos_names.append(pos_name)
        if not arrays:
            raise ValueError("No images found in tiff file.")
        if len(arrays) == 1:
            return arrays[0]
        else:
            stacked = xr.concat(arrays, dim='P')
            stacked = stacked.assign_coords(P=pos_names)
            return stacked
def open_any_format_to_xarray(file_path, dims=None, dask=False):
    """
    Reads an image or video file and returns an xarray.DataArray or xarray.Dataset
    with dimension names inferred from the metadata. Supports Dask arrays if dask=True.

    Args:
        file_path (str): Path to the image or video file.
        dims (tuple, optional): If provided, will override the inferred dimension names.
        dask (bool): If True, returns a Dask array instead of a NumPy array.

    Returns:
        xarray.DataArray or xarray.Dataset: The image data as an xarray object.
    """
    if file_path.endswith('.nd2'):
        return read_nd2_to_xarray_with_metadata(file_path, dims=dims, dask=dask)
    elif file_path.endswith('.lif'):
        return read_lif_to_xarray_with_metadata(file_path, dims=dims, dask=dask)
    elif file_path.endswith(('.tif', '.tiff')):
        return read_tiff_to_xarray_with_metadata(file_path, dims=dims, dask=dask)
    else:
        raise ValueError("Unsupported file format for xarray conversion.")
       
def read_nd2_to_xarray_with_metadata(file_path, dims=None, dask=False):
    """
    Reads an ND2 file and returns an xarray.DataArray with dimension names
    inferred from the metadata. Supports Dask arrays if dask=True.

    Args:
        file_path (str): Path to the ND2 file.
        dims (tuple, optional): If provided, will override the inferred dimension names.
        dask (bool): If True, returns a Dask array instead of a NumPy array.
    """
    image = nd2.imread(file_path, xarray=True, dask=dask)
    if dims is not None and len(dims) == image.ndim:
        coords = {d: np.arange(s) for d, s in zip(dims, image.shape)}
        return xr.DataArray(image.data, dims=dims, coords=coords)
    else:
        return image
    
def read_lif_to_numpy(file_path):
    if file_path.endswith('.lif'):
        new = LifFile(file_path)
        images = []
        names = []
        for image in new.get_iter_image():
            channels = [np.asarray(channel) for channel in image.get_iter_c()]
            images.append(np.asarray(channels))
            names.append(image.name)
        return images, names

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
        if 'C' in image.dims:
            try:
                image = image.sel({'C':'Trans'}).squeeze()
                messagebox.showinfo("Info", "Multiple channels detected. Only the 'Trans' channel will be processed.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process channels: {e}")
        if image.ndim == 2:
            # If the image is 2D, return it as a list with one element
            return [[np.array(image)]]
        elif 'P' in image.dims and image.ndim == 3:
            # If the image is 3D with a 'P' dimension, convert it to a list of 2D arrays
            return [[np.array(image.sel({'P':p})) for p in image.coords['P']]]
        elif 'T' in image.dims and image.ndim == 3:
            # If the image has a time dimension, convert it to a list of 2D arrays
            return [[np.array(image.sel({'T':t})) for t in image.coords['T']]]
        elif image.ndim == 4 and 'P' in image.dims and 'T' in image.dims:
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
    from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY
    frames_list = []

    # 1. Open the video file
    cap = VideoCapture(video_path)

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
        grayscale_frame = cvtColor(frame, COLOR_BGR2GRAY)

        # The 'grayscale_frame' is a NumPy ndarray
        frames_list.append(grayscale_frame)


    # 4. Release the VideoCapture object
    cap.release()


    return frames_list

def write_zarr(data, savedir, name):
    """
    Writes a NumPy array or Dask array to a Zarr file.
    """
    zarr_path = os.path.join(savedir, f"{name}.zarr")
    os.makedirs(zarr_path, exist_ok=True)  # Ensure the directory exists
    data.to_zarr(zarr_path, mode='w', consolidated=True)
    
if __name__=="__main__":
    def test_read_lif_to_numpy():
        import napari
        paths=filedialog.askopenfilenames(title="Select files")
        savedir = filedialog.askdirectory(title="Select save directory")
        viewer = napari.Viewer()
        for path in paths:
            images, names=read_lif_to_numpy(path)
            for image, name in zip(images, names):
                if 'A4' in name or 'Control' in name:
                    #viewer.add_image(data=image,name=name,channel_axis=0)
                    #Save image as npy
                    safe_name = name.replace('/', '_').replace('\\', '_')
                    np.save(os.path.join(savedir, f"{safe_name}.npy"), image)
        napari.run()
        print(names)

    def test_read_any_format_to_xarray_save_zarr():
        import napari
        paths=filedialog.askopenfilenames(title="Select files")
        savedir = filedialog.askdirectory(title="Select save directory")
        viewer = napari.Viewer()
        for path in paths:
            data = open_any_format_to_xarray(path, dask=True)
            if data is None:
                messagebox.showerror("Error", f"Failed to read file: {path}")
                continue
            # Extra processing if needed
            for image in data:
                viewer.add_image(image, name=image.attrs['name'], channel_axis=0)
            # Save as Zarr
                filename = os.path.splitext(os.path.basename(path))[0]+ f"_{image.attrs['name']}"
                write_zarr(image, savedir, filename)
        napari.run()
        print("Data saved to Zarr files in:", savedir)

    def test_read_aicsimage():
        #import napari
        paths = filedialog.askopenfilenames(title="Select files")
        #savedir = filedialog.askdirectory(title="Select save directory")
        #viewer = napari.Viewer()
        for path in paths:
            try:
                image = AICSImage(path)
                data = image.get_image_dask_data("CYX", S=0, T=0, Z=0)  # Adjust as needed
                if data is None:
                    messagebox.showerror("Error", f"Failed to read file: {path}")
                    continue
                #viewer.add_image(data, name=os.path.basename(path), channel_axis=0)
                #napari.run()
            except Exception as e:
                print(f"Error reading {path}: {e}")
            
    #test_read_any_format_to_xarray_save_zarr()
    test_read_aicsimage()