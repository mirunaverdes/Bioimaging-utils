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
from skimage.segmentation import clear_border
from skimage.exposure import rescale_intensity
from skimage.util import dtype_limits
from skimage.filters import gaussian, median
from pandas import DataFrame
from tqdm import tqdm
from readlif.reader import LifFile  
#import xarray as xr
from aicsimageio import AICSImage
from utils import crop_large_image, stitch_images, convert_to_minimal_format, namefile, find_channel_axis
#from skimage.restoration import gaussian_denoise, estimate_sigma
PROPERTIES_ALL = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std','bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')
PROPERTIES_MINIMAL = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std')
PROPERTIES_NO_INTENSITY = ('label', 'area', 'perimeter', 'centroid', 'bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')   
REGIONPROPS_FOR_TABLE = (
    # always ok (no intensity image needed)
    "label",
    "area",
    "bbox",
    "bbox_area",
    "centroid",
    "convex_area",
    "eccentricity",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "filled_area",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "local_centroid",
    "major_axis_length",
    "minor_axis_length",
    #"moments",
    #"moments_central",
    #"moments_hu",
    #"moments_normalized",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity",

    # require intensity_image=... when calling regionprops_table(...)
    "max_intensity",
    "mean_intensity",
    "min_intensity",
    #"weighted_centroid",
    #"weighted_local_centroid",
    #"moments_weighted",
    #"moments_weighted_central",
    #"moments_weighted_hu",
    #"moments_weighted_normalized",
)
def calculate_regionprops(masks, intensity_images, file_path, scene, spacing=(1, 1), 
                         channel_names=None, physical_pixel_sizes=None):
    """
    Calculate region properties for masks with intensity images.
    
    Args:
        masks (np.ndarray or list): 2D/3D array of segmentation masks or list of 2D masks
        intensity_images (list or np.ndarray): List of intensity images or single intensity image
        file_path (str): Path to the original file
        scene (str): Scene identifier
        spacing (tuple): Physical pixel spacing (Y, X)
        channel_names (list): List of channel names used for segmentation
        physical_pixel_sizes (object): Object containing X, Y pixel sizes
        
    Returns:
        pd.DataFrame: DataFrame containing region properties, or None if error occurred
    """
    from pandas import DataFrame, concat
    from skimage.measure import regionprops_table
    import numpy as np
    import os
    
    all_props = []
    physical_units = True
    
    # Define properties based on whether we have intensity data
    MORPHOLOGICAL_PROPS = (
        "label", "area", "bbox", "bbox_area", "centroid", "convex_area",
        "eccentricity", "equivalent_diameter_area", "euler_number", "extent",
        "feret_diameter_max", "filled_area", "inertia_tensor", "inertia_tensor_eigvals",
        "local_centroid", "major_axis_length", "minor_axis_length",
        "orientation", "perimeter", "perimeter_crofton", "solidity",
    )
    
    INTENSITY_PROPS = (
        "max_intensity", "mean_intensity", "min_intensity",
    )
    
    try:
        # Validate spacing
        if np.isnan(spacing).any() or np.any(np.array(spacing) <= 0) or spacing[0] is None or spacing[1] is None:
            print(f"Warning: Invalid spacing values {spacing}. Setting spacing to (1, 1).")
            spacing = (1, 1)
            physical_units = False
    except Exception as e:
        print(f"Error occurred while validating spacing: {e}")
        spacing = (1, 1)
        physical_units = False
    
    try:
        # Handle different mask formats - FIX HERE
        if isinstance(masks, list):
            # Already a list of 2D masks
            mask_list = masks
        elif hasattr(masks, 'ndim'):
            if masks.ndim == 3:
                # 3D array: convert to list of 2D masks
                mask_list = [masks[i] for i in range(masks.shape[0])]
            elif masks.ndim == 2:
                # Single 2D mask
                mask_list = [masks]
            else:
                raise ValueError(f"Unexpected mask dimensions: {masks.ndim}")
        else:
            raise ValueError(f"Masks must be numpy array or list, got {type(masks)}")
        
        # Handle intensity images
        if isinstance(intensity_images, list):
            intensity_list = intensity_images
        elif intensity_images is None:
            intensity_list = [None] * len(mask_list)
        elif hasattr(intensity_images, 'ndim'):
            if intensity_images.ndim == 4:
                # 4D array (T, C, Y, X): convert to list
                intensity_list = [intensity_images[i] for i in range(intensity_images.shape[0])]
            elif intensity_images.ndim == 3:
                # 3D array: could be (C, Y, X) or (T, Y, X)
                # Assume single timepoint with channels
                intensity_list = [intensity_images]
            elif intensity_images.ndim == 2:
                # Single 2D intensity image
                intensity_list = [intensity_images]
            else:
                raise ValueError(f"Unexpected intensity image dimensions: {intensity_images.ndim}")
        else:
            intensity_list = [intensity_images]
        
        # Ensure we have matching numbers of masks and intensity images
        if len(mask_list) != len(intensity_list):
            print(f"Warning: Number of masks ({len(mask_list)}) doesn't match number of intensity images ({len(intensity_list)})")
            min_length = min(len(mask_list), len(intensity_list))
            mask_list = mask_list[:min_length]
            intensity_list = intensity_list[:min_length]
        
        # Process each mask-intensity pair
        for i, (mask, intensity_img) in enumerate(zip(mask_list, intensity_list)):
            # Handle shape mismatches
            if intensity_img is not None:
                if intensity_img.ndim == 2 and mask.shape != intensity_img.shape:
                    print(f"Warning: Mask and intensity image have different shapes: {mask.shape} vs {intensity_img.shape}")
                    print("Cropping to minimum common dimensions")
                    min_shape = np.minimum(mask.shape, intensity_img.shape)
                    mask = mask[:min_shape[0], :min_shape[1]]
                    intensity_img = intensity_img[:min_shape[0], :min_shape[1]]
                elif intensity_img.ndim == 3 and mask.shape != intensity_img.shape[:2]:
                    print(f"Warning: Mask and multi-channel intensity image have different shapes: {mask.shape} vs {intensity_img.shape[:2]}")
                    print("Cropping to minimum common dimensions")
                    min_shape = np.minimum(mask.shape, intensity_img.shape[:2])
                    mask = mask[:min_shape[0], :min_shape[1]]
                    intensity_img = intensity_img[:min_shape[0], :min_shape[1]]
            
            # Select properties based on whether intensity image is available
            if intensity_img is not None:
                properties_to_use = MORPHOLOGICAL_PROPS + INTENSITY_PROPS
            else:
                properties_to_use = MORPHOLOGICAL_PROPS
                print(f"Warning: No intensity image for frame {i}, calculating morphological properties only")
            
            # Calculate regionprops
            props = regionprops_table(
                mask,
                intensity_image=intensity_img,  # Can be None
                properties=properties_to_use,
                spacing=spacing
            )
            
            # Convert to DataFrame and add metadata
            df = DataFrame(props)
            df['file'] = os.path.basename(file_path)
            df['scene'] = scene
            df['frame'] = i
            df['channel_names'] = str(channel_names) if channel_names else "Unknown"
            df['has_intensity_data'] = intensity_img is not None
            
            # Add pixel size information
            if physical_pixel_sizes:
                df['x_voxel_size'] = physical_pixel_sizes.X
                df['y_voxel_size'] = physical_pixel_sizes.Y
            else:
                df['x_voxel_size'] = spacing[1]
                df['y_voxel_size'] = spacing[0]
            
            df['physical_units_applied'] = physical_units
            
            all_props.append(df)
        
        # Concatenate all properties
        if all_props:
            props_df = concat(all_props, ignore_index=True)
            return props_df
        else:
            print("No region properties calculated")
            return None
            
    except Exception as e:
        print(f"Error occurred while calculating regionprops: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_regionprops(props_df, savedir, base_filename, scene=None):
    """
    Save region properties DataFrame to CSV file.
    
    Args:
        props_df (pd.DataFrame): DataFrame containing region properties
        savedir (str): Directory to save the CSV file
        base_filename (str): Base filename (without extension)
        scene (str, optional): Scene identifier to append to filename
        
    Returns:
        str: Path to saved file, or None if error occurred
    """
    import os
    
    if props_df is None or props_df.empty:
        print("No region properties to save")
        return None
    
    try:
        # Create filename
        if scene:
            reg_filename = f"{base_filename}_{scene}"
        else:
            reg_filename = base_filename
        
        # Check if file exists and increment if necessary
        counter = 1
        original_filename = reg_filename
        while os.path.exists(os.path.join(savedir, f"{reg_filename}.csv")):
            reg_filename = f"{original_filename}_{counter}"
            counter += 1
        
        # Save to CSV
        filepath = os.path.join(savedir, f"{reg_filename}.csv")
        props_df.to_csv(filepath, index=False)
        print(f"Region properties saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error occurred while saving regionprops: {e}")
        return None

    
def prepare_intensity_list(intensity_data, masks, chunksize, subset):
    """
    Prepare intensity images for regionprops calculation.
    
    Returns:
        list: List of intensity images with channels last (Y, X, C) or [None] if no intensity data
    """
    if chunksize != [1, 1] and subset != slice(None):
        # Chunked processing
        intensity_list = crop_large_image(intensity_data, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
        # Find channel axis
        channel_axis = find_channel_axis(intensity_list[0])
        # Move channels to last dimension
        intensity_list = [np.moveaxis(img, channel_axis, -1) for img in intensity_list]
        return intensity_list[subset]
    
    if intensity_data is None:
        # No intensity data available; raise warning and return None
        print("Warning: No intensity data available for regionprops calculation.")
        return None
        
    
    # Handle intensity data
    try:
        if intensity_data.ndim == 4:  # (T, C, Y, X)
            channel_axis = find_channel_axis(intensity_data[0])
            intensity_list = [np.moveaxis(intensity_data[t], channel_axis, -1) for t in range(intensity_data.shape[0])]
        elif intensity_data.ndim == 3:  # (C, Y, X) - single timepoint
            intensity_list = [np.moveaxis(intensity_data, 0, -1)]
        else:
            print(f"Warning: Unexpected intensity data shape: {intensity_data.shape}")
            intensity_list = None

        return intensity_list
        
    except Exception as e:
        print(f"Error preparing intensity list: {e}")
        # Fallback to None
        return None
    
def prepare_masks_list(masks, chunksize, subset):
    """
    Prepare masks for regionprops calculation.
    
    Returns:
        list: List of masks
    """
    if masks is None:
        print("Warning: No masks available for regionprops calculation.")
        return None

    if chunksize != [1, 1] and subset != slice(None):
        # Chunked processing
        masks_list = crop_large_image(masks, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
        return masks_list[subset]
    
    if masks.ndim == 3:  # (T, Y, X)
        return [masks[t] for t in range(masks.shape[0])]
    elif masks.ndim == 2:  # (Y, X) - single mask
        return [masks]
    else:
        print(f"Warning: Unexpected masks shape: {masks.shape}")
        return None

def check_masks_intensity_shapes(masks_list, intensity_list):
    """
    Check if the shapes of masks and intensity images match and try to fix if not.
    
    Args:
        masks_list (list): List of mask images.
        intensity_list (list): List of intensity images (channels last).

    Returns:
        tuple: (bool, fixed_masks_list, fixed_intensity_list) - 
               True if shapes match/were fixed, False otherwise.
               Returns the potentially fixed lists.
    """
    if masks_list is None or intensity_list is None:
        print("Warning: One or both lists are None.")
        return False, masks_list, intensity_list

    if len(masks_list) != len(intensity_list):
        print(f"Warning: Lengths of masks ({len(masks_list)}) and intensity ({len(intensity_list)}) lists do not match.")
        # Try to fix by truncating to minimum length
        min_length = min(len(masks_list), len(intensity_list))
        masks_list = masks_list[:min_length]
        intensity_list = intensity_list[:min_length]
        print(f"Truncated both lists to length {min_length}")
    
    fixed_masks = []
    fixed_intensity = []
    all_shapes_match = True
    
    for i, (mask, intensity) in enumerate(zip(masks_list, intensity_list)):
        if intensity is None:
            # If intensity is None, just keep the mask as is
            fixed_masks.append(mask)
            fixed_intensity.append(intensity)
            continue
            
        # Get expected shape from intensity image
        if intensity.ndim == 3:  # (Y, X, C) - channels last
            expected_shape = intensity.shape[:2]  # (Y, X)
        elif intensity.ndim == 2:  # (Y, X) - single channel
            expected_shape = intensity.shape
        else:
            print(f"Warning: Unexpected intensity image dimensions at index {i}: {intensity.shape}")
            fixed_masks.append(mask)
            fixed_intensity.append(intensity)
            all_shapes_match = False
            continue
        
        # Check if mask shape matches expected shape
        if mask.shape != expected_shape:
            print(f"Warning: Shape mismatch at index {i}: mask {mask.shape} vs expected {expected_shape}")
            
            # Try to fix by cropping to minimum common dimensions
            try:
                min_shape = tuple(min(m, e) for m, e in zip(mask.shape, expected_shape))
                
                # Crop mask
                fixed_mask = mask[:min_shape[0], :min_shape[1]]
                
                # Crop intensity image
                if intensity.ndim == 3:
                    fixed_intensity_img = intensity[:min_shape[0], :min_shape[1], :]
                else:
                    fixed_intensity_img = intensity[:min_shape[0], :min_shape[1]]
                
                print(f"Fixed shapes at index {i}: mask {mask.shape} -> {fixed_mask.shape}, "
                      f"intensity {intensity.shape} -> {fixed_intensity_img.shape}")
                
                fixed_masks.append(fixed_mask)
                fixed_intensity.append(fixed_intensity_img)
                
            except Exception as e:
                print(f"Failed to fix shapes at index {i}: {e}")
                # Keep original shapes if fixing fails
                fixed_masks.append(mask)
                fixed_intensity.append(intensity)
                all_shapes_match = False
        else:
            # Shapes match, keep as is
            fixed_masks.append(mask)
            fixed_intensity.append(intensity)
    
    return all_shapes_match, fixed_masks, fixed_intensity


def check_and_fix_shapes(masks_list, intensity_list):
    """
    Wrapper function that checks and fixes shapes, returning the corrected lists.
    
    Args:
        masks_list (list): List of mask images
        intensity_list (list): List of intensity images
        
    Returns:
        tuple: (masks_list, intensity_list) - corrected lists
    """
    shapes_ok, fixed_masks, fixed_intensity = check_masks_intensity_shapes(masks_list, intensity_list)
    
    if not shapes_ok:
        print("Some shape mismatches were detected and attempted to be fixed.")
    else:
        print("All mask and intensity image shapes are compatible.")
    
    return fixed_masks, fixed_intensity


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


@timeit
def process_files_multichannel_2(
    files, model, model_diameter, savedir,
    get_regionprops=True, segment_channels=[1, 0], intensity_channels=None, 
    subset=slice(None), chunksize=[1, 1], scene_identifiers=None, time_subset=slice(None)
):
    """
    Processes a list of image files (2D multichannel timelapses) readable with AICSImage, segments them using the Cellpose model,
    and saves the masks and region properties to the specified directory.
    Args:
        files (list): List of file paths to process.
        model (CellposeModel): The Cellpose model to use for segmentation.
        model_diameter (int): The diameter of the cells for the model.
        savedir (str): Directory where the masks and region properties will be saved.
        get_regionprops (bool): Whether to compute and save region properties.
        segment_channels (list): List of channel indices to use for segmentation. First index is the channel to segment, the second is optional channel with nuclei.
        intensity_channels (list): List of channel indices to use for regionprops intensity metrics.
        subset (slice): Slice object to select a subset of images from the cropped image list. Is slice(None) by default, which means all images will be processed.
        chunksize (list): List of chunk sizes for processing the images. The first element is the chunk size for the x dimension, the second element is the chunk size for the y dimension.
        scene_identifiers (list): List of scene identifiers to process. If None, all scenes will be processed.
        time_subset (slice): Slice object to select a subset of time points. Is slice(None) by default, which means all time points will be processed.
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
                data = np.load(file_path)
                img = AICSImage(data)
        
        # Check if the image has multiple scenes
        if len(img.scenes) == 0:
            print(f"No scenes found in {file_path}. Skipping this file.")
            continue
            
        scenes = img.scenes if scene_identifiers is None else [scene for scene in img.scenes if any(s in scene for s in scene_identifiers)]
        
        # Initialize list to collect all regionprops from all scenes for this file
        all_scenes_props = []
        
        for scene in scenes:
            img.set_scene(scene)
            
            # Create the filepaths for saving
            base = os.path.basename(file_path).split('.')[0]
            masks_filepath = namefile(savedir, base, prefix=f"masks-{model_diameter}", suffix=scene, ext="npy")

            # Get ALL channel data at once
            full_img = img.get_image_data("TCYX")  # Single read
            
            # Use views/slices instead of copying
            seg_data = full_img[:, segment_channels]  # View, not copy
            intensity_data = full_img[:, intensity_channels] if intensity_channels else None
            
            # Convert to list for Cellpose
            seg_list = [seg_data[t] for t in range(seg_data.shape[0])]  # Views of time points
            
            # Only chunk if the image is not a timelapse
            if chunksize != [1, 1] and len(seg_list) == 1:
                print(f"Processing image in chunks of size {chunksize}...")
                seg_list = crop_large_image(seg_data, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                try:
                    seg_list = seg_list[subset]
                except Exception as e:
                    print(f"Error occurred while applying subset: {e}")

            # Segment
            masks, _, _ = model.eval(seg_list, # this can now be a list of 2D or 3D multichannel images
                                     #channels = segment_channels, 
                                     diameter=model_diameter)
            
            
            # Remove masks touching edges
            masks = [clear_border(mask) for mask in masks] if isinstance(masks, list) else clear_border(masks)
           
            # Save masks
            if chunksize != [1, 1]:
                # If no subset is specified, and image was tiled, stitch all masks together
                masks = stitch_images(masks, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                # Relabel masks to avoid duplicate labels after stitching
                masks = label(masks)
            
            masks = convert_to_minimal_format(masks)  # Convert to minimal format  
            # To fix masks file doesn't save properly- potentially due to file renaming              
            np.save(masks_filepath, masks) 
            # Prepare masks for regionprops
            masks_list = [masks[t] for t in range(masks.shape[0])] if masks.ndim == 3 else [masks]
            # Prepare intensity images for regionprops
            if chunksize != [1, 1] and subset != slice(None):
                intensity_list = crop_large_image(intensity_data, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                intensity_list = intensity_list[subset]
            else:
                intensity_list = [np.moveaxis(intensity_data[t],0,-1) for t in range(intensity_data.shape[0])] if intensity_data is not None else [None] * len(masks)
    
            # Calculate regionprops for this scene using the separate function
            if get_regionprops:
                try:
                    # Get spacing information
                    spacing = (img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X)
                    masks_list, intensity_list = check_and_fix_shapes(masks_list, intensity_list)
                    # Calculate regionprops for this scene
                    scene_props_df = calculate_regionprops(
                        masks=masks_list,
                        intensity_images=intensity_list,
                        file_path=file_path,
                        scene=scene,
                        spacing=spacing,
                        channel_names=[img.channel_names[s] for s in segment_channels],
                        physical_pixel_sizes=img.physical_pixel_sizes
                    )
                    
                    # Add to collection if successful
                    if scene_props_df is not None:
                        all_scenes_props.append(scene_props_df)
                    
                except Exception as e:
                    print(f"Error in regionprops calculation for scene {scene}: {e}")
        
        # Save combined regionprops for all scenes of this file
        if get_regionprops and all_scenes_props:
            try:
                # Combine all scenes for this file
                combined_props_df = concat(all_scenes_props, ignore_index=True)
                
                # Save using the separate function
                save_regionprops(combined_props_df, savedir, base)
                
            except Exception as e:
                print(f"Error saving combined regionprops for file {file_path}: {e}")

@timeit
def process_files_multichannel_3(
    files, model, model_diameter, savedir,
    get_regionprops=True, get_tracking=True, segment_channels=[1, 0], intensity_channels=None, 
    subset=slice(None), chunksize=[1, 1], scene_identifiers=None, time_subset=slice(None)
):
    """
    Processes a list of image files (2D multichannel timelapses) readable with AICSImage, segments them using the Cellpose model,
    tracks them using Trackastra, and saves the masks, tracking data, and region properties to the specified directory.
    
    Args:
        files (list): List of file paths to process.
        model (CellposeModel): The Cellpose model to use for segmentation.
        model_diameter (int): The diameter of the cells for the model.
        savedir (str): Directory where the masks, tracking data, and region properties will be saved.
        get_regionprops (bool): Whether to compute and save region properties.
        get_tracking (bool): Whether to perform cell tracking using Trackastra.
        segment_channels (list): List of channel indices to use for segmentation.
        intensity_channels (list): List of channel indices to use for regionprops intensity metrics.
        subset (slice): Slice object to select a subset of images from the cropped image list.
        chunksize (list): List of chunk sizes for processing the images.
        scene_identifiers (list): List of scene identifiers to process. If None, all scenes will be processed.
        time_subset (slice): Slice object to select a subset of time points.
    """
    # Import tracking functions
    from tracking import track_Trackastra, preprocess_masks
    import pickle
    
    for file_path in tqdm(files, desc="Processing files", unit="file"):
        
        try:
            img = AICSImage(file_path)
        except:
            if file_path.endswith(".npy"):
                data = np.load(file_path)
                img = AICSImage(data)
        
        # Check if the image has multiple scenes
        if len(img.scenes) == 0:
            print(f"No scenes found in {file_path}. Skipping this file.")
            continue
            
        scenes = img.scenes if scene_identifiers is None else [scene for scene in img.scenes if any(s in scene for s in scene_identifiers)]
        
        # Initialize list to collect all regionprops from all scenes for this file
        all_scenes_props = []
        
        for scene in scenes:
            img.set_scene(scene)
            
            # Create the filepaths for saving
            base = os.path.basename(file_path).split('.')[0]
            masks_filepath = namefile(savedir, base, prefix=f"masks-{model_diameter}", suffix=scene, ext="npy")
            image_filepath = namefile(savedir, base, prefix="image", suffix=scene, ext="npy")

            # Get ALL channel data at once
            full_img = img.get_image_data("TCYX")  # Single read
            
            # Use views/slices instead of copying
            seg_data = full_img[:, segment_channels]  # View, not copy
            intensity_data = full_img[:, intensity_channels] if intensity_channels else None
            
            # Convert to list for Cellpose
            seg_list = [seg_data[t] for t in range(seg_data.shape[0])]  # Views of time points
            
            # Only chunk if the image is not a timelapse
            if chunksize != [1, 1] and len(seg_list) == 1:
                print(f"Processing image in chunks of size {chunksize}...")
                seg_list = crop_large_image(seg_data, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                try:
                    seg_list = seg_list[subset]
                except Exception as e:
                    print(f"Error occurred while applying subset: {e}")

            # Segment
            print(f"🔬 Segmenting {len(seg_list)} frames...")
            masks, _, _ = model.eval(
                seg_list,
                diameter=model_diameter
            )
            
            # Remove masks touching edges
            masks = [clear_border(mask) for mask in masks] if isinstance(masks, list) else clear_border(masks)
           
            # Handle chunked processing for masks
            if chunksize != [1, 1]:
                # If no subset is specified, and image was tiled, stitch all masks together
                masks = stitch_images(masks, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                # Relabel masks to avoid duplicate labels after stitching
                masks = label(masks)
            
            # Convert to minimal format and ensure it's a proper array
            masks = convert_to_minimal_format(masks)
            
            # Save original masks
            np.save(masks_filepath, masks)
            
            # Prepare images for tracking (convert to list if needed)
            if isinstance(seg_list[0], np.ndarray) and seg_list[0].ndim == 3:
                # Multi-channel images - take first channel for tracking
                images_for_tracking = [img_frame[0] for img_frame in seg_list]
            else:
                # Single channel images
                images_for_tracking = seg_list
            
            # Save images for tracking
            np.save(image_filepath, np.array(images_for_tracking))
            
            # Prepare masks for tracking
            if masks.ndim == 3:
                masks_for_tracking = [masks[t] for t in range(masks.shape[0])]
            else:
                masks_for_tracking = [masks]
            
            # Tracking
            if get_tracking and len(masks_for_tracking) > 1:
                try:
                    print(f"🔗 Tracking {len(masks_for_tracking)} frames...")
                    
                    # Preprocess masks for tracking
                    masks_preprocessed = preprocess_masks(masks_for_tracking)
                    
                    # Perform tracking
                    track_graph, ctc_tracks, masks_tracked, napari_tracks = track_Trackastra(
                        images_for_tracking, 
                        masks_preprocessed
                    )
                    
                    # Save tracking results
                    tracking_base = f"tracked_{base}_{scene}"
                    
                    # Save track graph
                    with open(namefile(savedir,tracking_base,'','graph','pkl'), 'wb') as f:
                        pickle.dump(track_graph, f)
                    
                    # Save CTC tracks
                    ctc_tracks.to_csv(namefile(savedir,tracking_base,'','ctc','csv'), index=False)

                    # Save tracked masks
                    np.save(namefile(savedir,tracking_base,'','masks','npy'), masks_tracked)

                    # Save napari tracks
                    #np.save(namefile(savedir,tracking_base,'','napari_tracks','npy'), napari_tracks)

                    print(f"✓ Tracking completed and saved for scene {scene}")
                    
                    # Use tracked masks for regionprops if tracking was successful
                    final_masks = masks_tracked
                    
                except Exception as e:
                    print(f"❌ Tracking failed for scene {scene}: {e}")
                    print("Using original masks for regionprops...")
                    final_masks = masks
            else:
                print("⏭️ Skipping tracking (disabled or single frame)")
                final_masks = masks
            
            # Calculate regionprops using final masks (tracked or original)
            if get_regionprops:
                try:
                    print(f"📊 Calculating region properties...")
                    
                    # Get spacing information
                    spacing = (img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X)
                    
                    # Prepare masks for regionprops
                    if final_masks.ndim == 3:
                        masks_list = [final_masks[t] for t in range(final_masks.shape[0])]
                    else:
                        masks_list = [final_masks]
                    
                    # Prepare intensity images for regionprops
                    if chunksize != [1, 1] and subset != slice(None):
                        intensity_list = crop_large_image(intensity_data, n_segments_x=chunksize[0], n_segments_y=chunksize[1])
                        intensity_list = intensity_list[subset]
                    else:
                        if intensity_data is not None:
                            intensity_list = [np.moveaxis(intensity_data[t], 0, -1) for t in range(intensity_data.shape[0])]
                        else:
                            intensity_list = [None] * len(masks_list)
                    
                    # Check and fix shapes
                    masks_list, intensity_list = check_and_fix_shapes(masks_list, intensity_list)
                    
                    # Calculate regionprops for this scene
                    scene_props_df = calculate_regionprops(
                        masks=masks_list,
                        intensity_images=intensity_list,
                        file_path=file_path,
                        scene=scene,
                        spacing=spacing,
                        channel_names=[img.channel_names[s] for s in segment_channels],
                        physical_pixel_sizes=img.physical_pixel_sizes
                    )
                    
                    # Add tracking information to regionprops if available
                    if get_tracking and 'ctc_tracks' in locals():
                        scene_props_df['has_tracking_data'] = True
                        # You could merge tracking IDs here if needed
                        # scene_props_df = merge_tracking_regionprops(scene_props_df, ctc_tracks)
                    else:
                        scene_props_df['has_tracking_data'] = False
                    
                    # Add to collection if successful
                    if scene_props_df is not None:
                        all_scenes_props.append(scene_props_df)
                        print(f"✓ Region properties calculated for scene {scene}")
                    
                except Exception as e:
                    print(f"❌ Error in regionprops calculation for scene {scene}: {e}")
        
        # Save combined regionprops for all scenes of this file
        if get_regionprops and all_scenes_props:
            try:
                # Combine all scenes for this file
                combined_props_df = concat(all_scenes_props, ignore_index=True)
                
                # Save using the separate function
                save_regionprops(combined_props_df, savedir, base)
                print(f"✓ Combined region properties saved for {base}")
                
            except Exception as e:
                print(f"❌ Error saving combined regionprops for file {file_path}: {e}")

        print(f"🎉 Processing completed for {os.path.basename(file_path)}")


def merge_tracking_regionprops(regionprops_df, ctc_tracks):
    """
    Helper function to merge tracking information with regionprops data.
    
    Args:
        regionprops_df (pd.DataFrame): DataFrame containing region properties
        ctc_tracks (pd.DataFrame): DataFrame containing tracking information
        
    Returns:
        pd.DataFrame: Merged DataFrame with tracking IDs
    """
    try:
        # Merge based on frame and label (this might need adjustment based on your data structure)
        # This is a simplified merge - you may need to adapt based on your specific data format
        merged_df = regionprops_df.merge(
            ctc_tracks[['frame', 'label', 'track_id']], 
            left_on=['frame', 'label'], 
            right_on=['frame', 'label'], 
            how='left'
        )
        return merged_df
    except Exception as e:
        print(f"Warning: Could not merge tracking data with regionprops: {e}")
        return regionprops_df

def user_interface():
    import customtkinter as ctk
    from tkinter import filedialog as fd
    from tkinter import messagebox
    
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    
    app = ctk.CTk()
    app.geometry("1000x900")  # Increased height for tracking section
    app.title("🌭 dashUND Segmentation & Tracking")
    app.minsize(800, 700)
    
    # Configure grid weights for responsive design
    app.grid_columnconfigure(0, weight=1)
    app.grid_rowconfigure(0, weight=1)
    
    # Create main scrollable frame
    main_scrollable = ctk.CTkScrollableFrame(app)
    main_scrollable.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    main_scrollable.grid_columnconfigure(0, weight=1)
    
    # Title with better styling
    title_label = ctk.CTkLabel(
        main_scrollable, 
        text="🌭 dashUND Segmentation & Tracking", 
        font=ctk.CTkFont(size=28, weight="bold")
    )
    title_label.grid(row=0, column=0, pady=(10, 20))
    
    # Status label for user feedback
    status_label = ctk.CTkLabel(
        main_scrollable, 
        text="Select files to begin", 
        font=ctk.CTkFont(size=12),
        text_color="gray"
    )
    status_label.grid(row=1, column=0, pady=(0, 20))
    
    # Variables to store user inputs
    file_checkboxes = []
    selected_channel_var = ctk.StringVar()
    intensity_channels_vars = []
    model_diameter_var = ctk.IntVar(value=30)
    get_regionprops_var = ctk.BooleanVar(value=True)
    get_tracking_var = ctk.BooleanVar(value=False)  # New tracking variable
    custom_model_var = ctk.BooleanVar(value=False)
    model_path_var = ctk.StringVar()
    savedir_var = ctk.StringVar()
    scene_identifiers_var = ctk.StringVar()
    
    # Helper function to update status
    def update_status(message, color="gray"):
        status_label.configure(text=message, text_color=color)
        app.update()
    
    # File Selection Section
    file_section = ctk.CTkFrame(main_scrollable, corner_radius=15)
    file_section.grid(row=2, column=0, padx=20, pady=10, sticky="ew", ipadx=20, ipady=15)
    file_section.grid_columnconfigure(0, weight=1)
    
    file_label = ctk.CTkLabel(
        file_section, 
        text="📁 File Selection", 
        font=ctk.CTkFont(size=20, weight="bold")
    )
    file_label.grid(row=0, column=0, pady=(0, 15))
    
    # File selection buttons in a horizontal layout
    file_buttons_frame = ctk.CTkFrame(file_section, fg_color="transparent")
    file_buttons_frame.grid(row=1, column=0, sticky="ew", padx=10)
    file_buttons_frame.grid_columnconfigure((0, 1), weight=1)
    
    def browse_files():
        update_status("Browsing for files...", "blue")
        files = fd.askopenfilenames(
            title="Select image files",
            filetypes=[
                ("All supported", "*.tif *.tiff *.nd2 *.lif *.czi *.png *.jpg"),
                ("TIFF files", "*.tif *.tiff"),
                ("ND2 files", "*.nd2"),
                ("LIF files", "*.lif"),
                ("CZI files", "*.czi"),
                ("All files", "*.*")
            ]
        )
        if files:
            # Clear previous checkboxes
            for widget in file_list_frame.winfo_children():
                widget.destroy()
            file_checkboxes.clear()
            
            for file in files:
                var = ctk.BooleanVar(value=True)
                chk = ctk.CTkCheckBox(
                    file_list_frame, 
                    text=os.path.basename(file), 
                    variable=var,
                    font=ctk.CTkFont(size=11)
                )
                chk.pack(anchor="w", padx=5, pady=3)
                file_checkboxes.append((file, var))
            
            update_status(f"✓ {len(files)} files selected", "green")
            
            # Update channel options based on first file
            update_channel_options(files[0])
            
            # Enable process button check
            check_ready_to_process()
        else:
            update_status("No files selected", "orange")
    
    def browse_savedir():
        update_status("Selecting save directory...", "blue")
        directory = fd.askdirectory(title="Select save directory")
        if directory:
            savedir_var.set(directory)
            # Truncate long paths for display
            display_path = os.path.basename(directory) if len(directory) < 50 else f"...{directory[-40:]}"

            savedir_label.configure(text=f"💾 Save to: {display_path}")
            update_status("✓ Save directory selected", "green")
            check_ready_to_process()
        else:
            update_status("No save directory selected", "orange")
    
    def browse_model():
        model_file = fd.askopenfilename(
            title="Select Cellpose model", 
            filetypes=[("Model files", "*.pkl *.pth"), ("All files", "*.*")]
        )
        if model_file:
            model_path_var.set(model_file)
            model_name = os.path.basename(model_file)
            model_path_label.configure(text=f"🧠 Model: {model_name}")
    
    browse_button = ctk.CTkButton(
        file_buttons_frame, 
        text="📂 Browse Files", 
        command=browse_files,
        height=35,
        font=ctk.CTkFont(size=13, weight="bold")
    )
    browse_button.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
    
    savedir_button = ctk.CTkButton(
        file_buttons_frame, 
        text="📁 Save Directory", 
        command=browse_savedir,
        height=35,
        font=ctk.CTkFont(size=13, weight="bold")
    )
    savedir_button.grid(row=0, column=1, padx=(10, 0), pady=5, sticky="ew")
    
    # File list with better styling
    file_list_frame = ctk.CTkScrollableFrame(file_section, height=120, corner_radius=10)
    file_list_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
    
    # Save directory display with better styling
    savedir_label = ctk.CTkLabel(
        file_section, 
        text="💾 Save directory: Not selected", 
        font=ctk.CTkFont(size=12),
        text_color="orange"
    )
    savedir_label.grid(row=3, column=0, pady=5)
    
    # Model Configuration Section
    model_section = ctk.CTkFrame(main_scrollable, corner_radius=15)
    model_section.grid(row=3, column=0, padx=20, pady=10, sticky="ew", ipadx=20, ipady=15)
    model_section.grid_columnconfigure((0, 1), weight=1)
    
    model_label = ctk.CTkLabel(
        model_section, 
        text="🧠 Model Configuration", 
        font=ctk.CTkFont(size=20, weight="bold")
    )
    model_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
    
    # Model diameter with better layout
    diameter_frame = ctk.CTkFrame(model_section, fg_color="transparent")
    diameter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    diameter_frame.grid_columnconfigure(1, weight=1)
    
    diameter_label = ctk.CTkLabel(diameter_frame, text="🔍 Cell Diameter (pixels):")
    diameter_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
    
    diameter_entry = ctk.CTkEntry(
        diameter_frame, 
        textvariable=model_diameter_var, 
        width=100,
        placeholder_text="30"
    )
    diameter_entry.grid(row=0, column=1, sticky="w")
    
    # Custom model section
    custom_model_frame = ctk.CTkFrame(model_section, fg_color="transparent")
    custom_model_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
    custom_model_frame.grid_columnconfigure(1, weight=1)
    
    def toggle_custom_model():
        state = "normal" if custom_model_var.get() else "disabled"
        model_path_button.configure(state=state)
        if not custom_model_var.get():
            model_path_var.set("")
            model_path_label.configure(text="🧠 Model: Default Cellpose")
    
    custom_model_check = ctk.CTkCheckBox(
        custom_model_frame, 
        text="Use custom model", 
        variable=custom_model_var, 
        command=toggle_custom_model
    )
    custom_model_check.grid(row=0, column=0, sticky="w", padx=(0, 10))
    
    model_path_button = ctk.CTkButton(
        custom_model_frame, 
        text="Select Model", 
        command=browse_model, 
        state="disabled",
        width=120
    )
    model_path_button.grid(row=0, column=1, sticky="w")
    
    model_path_label = ctk.CTkLabel(
        model_section, 
        text="🧠 Model: Default Cellpose", 
        font=ctk.CTkFont(size=12)
    )
    model_path_label.grid(row=3, column=0, columnspan=2, pady=5)
    
    # Analysis Options Section
    analysis_section = ctk.CTkFrame(main_scrollable, corner_radius=15)
    analysis_section.grid(row=4, column=0, padx=20, pady=10, sticky="ew", ipadx=20, ipady=15)
    analysis_section.grid_columnconfigure(0, weight=1)
    
    analysis_label = ctk.CTkLabel(
        analysis_section, 
        text="📊 Analysis Options", 
        font=ctk.CTkFont(size=20, weight="bold")
    )
    analysis_label.grid(row=0, column=0, pady=(0, 15))
    
    # Analysis options frame
    analysis_options_frame = ctk.CTkFrame(analysis_section, fg_color="transparent")
    analysis_options_frame.grid(row=1, column=0, sticky="ew", padx=10)
    analysis_options_frame.grid_columnconfigure((0, 1), weight=1)
    
    # Regionprops checkbox
    regionprops_check = ctk.CTkCheckBox(
        analysis_options_frame, 
        text="📈 Calculate region properties", 
        variable=get_regionprops_var,
        font=ctk.CTkFont(size=13)
    )
    regionprops_check.grid(row=0, column=0, sticky="w", padx=10, pady=10)
    
    # Tracking checkbox
    tracking_check = ctk.CTkCheckBox(
        analysis_options_frame, 
        text="🔗 Perform cell tracking", 
        variable=get_tracking_var,
        font=ctk.CTkFont(size=13)
    )
    tracking_check.grid(row=0, column=1, sticky="w", padx=10, pady=10)
    
    # Tracking info label
    tracking_info_label = ctk.CTkLabel(
        analysis_section, 
        text="ℹ️ Tracking requires multi-frame time-lapse data. Uses Trackastra for linking cells across frames.", 
        font=ctk.CTkFont(size=11, slant="italic"),
        text_color="gray",
        wraplength=800
    )
    tracking_info_label.grid(row=2, column=0, pady=5)
    
    # Channel Selection Section
    channel_section = ctk.CTkFrame(main_scrollable, corner_radius=15)
    channel_section.grid(row=5, column=0, padx=20, pady=10, sticky="ew", ipadx=20, ipady=15)
    channel_section.grid_columnconfigure(0, weight=1)
    
    channel_label = ctk.CTkLabel(
        channel_section, 
        text="🎨 Channel Configuration", 
        font=ctk.CTkFont(size=20, weight="bold")
    )
    channel_label.grid(row=0, column=0, pady=(0, 15))
    
    # Segmentation channel frame
    seg_channel_frame = ctk.CTkFrame(channel_section, corner_radius=10)
    seg_channel_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    seg_channel_frame.grid_columnconfigure(0, weight=1)
    
    seg_channel_label = ctk.CTkLabel(
        seg_channel_frame, 
        text="🎯 Segmentation Channel:", 
        font=ctk.CTkFont(size=16, weight="bold")
    )
    seg_channel_label.grid(row=0, column=0, pady=8)
    
    seg_channel_container = ctk.CTkFrame(seg_channel_frame, fg_color="transparent")
    seg_channel_container.grid(row=1, column=0, pady=8)
    
    # Intensity channels frame
    int_channel_frame = ctk.CTkFrame(channel_section, corner_radius=10)
    int_channel_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
    int_channel_frame.grid_columnconfigure(0, weight=1)
    
    int_channel_label = ctk.CTkLabel(
        int_channel_frame, 
        text="📈 Intensity Channels:", 
        font=ctk.CTkFont(size=16, weight="bold")
    )
    int_channel_label.grid(row=0, column=0, pady=8)
    
    int_channel_container = ctk.CTkFrame(int_channel_frame, fg_color="transparent")
    int_channel_container.grid(row=1, column=0, pady=8)
    
    # Scene Selection Section
    scene_section = ctk.CTkFrame(main_scrollable, corner_radius=15)
    scene_section.grid(row=6, column=0, padx=20, pady=10, sticky="ew", ipadx=20, ipady=15)
    scene_section.grid_columnconfigure(0, weight=1)
    
    scene_label = ctk.CTkLabel(
        scene_section, 
        text="🎬 Scene Selection", 
        font=ctk.CTkFont(size=20, weight="bold")
    )
    scene_label.grid(row=0, column=0, pady=(0, 15))
    
    scene_info_label = ctk.CTkLabel(
        scene_section, 
        text="Scene identifiers (comma-separated, leave empty for all):", 
        font=ctk.CTkFont(size=13)
    )
    scene_info_label.grid(row=1, column=0, pady=5)
    
    scene_example_label = ctk.CTkLabel(
        scene_section, 
        text="Select files to see available scenes", 
        font=ctk.CTkFont(size=11, slant="italic"),
        text_color="gray"
    )
    scene_example_label.grid(row=2, column=0, pady=5)
    
    scene_entry = ctk.CTkEntry(
        scene_section, 
        textvariable=scene_identifiers_var, 
        width=400,
        placeholder_text="e.g., Scene1, Scene2 (leave empty for all)"
    )
    scene_entry.grid(row=3, column=0, pady=10)
    
    def update_channel_options(file_path):
        try:
            update_status("Reading file metadata...", "blue")
            image = AICSImage(file_path)
            channels = image.channel_names
            scenes = image.scenes
            
            # Check if it's a time-lapse for tracking
            time_points = image.dims.T
            if time_points > 1:
                tracking_info_label.configure(
                    text=f"ℹ️ Time-lapse detected: {time_points} frames. Tracking is available for this dataset.",
                    text_color="green"
                )
                tracking_check.configure(state="normal")
            else:
                tracking_info_label.configure(
                    text="ℹ️ Single time-point detected. Tracking requires multi-frame time-lapse data.",
                    text_color="orange"
                )
                tracking_check.configure(state="disabled")
                get_tracking_var.set(False)
            
            # Clear previous channel widgets
            for widget in seg_channel_container.winfo_children():
                widget.destroy()
            for widget in int_channel_container.winfo_children():
                widget.destroy()
            intensity_channels_vars.clear()
            
            # Update selected channel variable
            selected_channel_var.set("0")  # Default to first channel
            
            # Create segmentation channel radio buttons
            for i, channel in enumerate(channels):
                radio = ctk.CTkRadioButton(
                    seg_channel_container, 
                    text=channel, 
                    variable=selected_channel_var, 
                    value=str(i),
                    font=ctk.CTkFont(size=12)
                )
                radio.pack(side="left", padx=8, pady=5)
            
            # Create intensity channel checkboxes
            for i, channel in enumerate(channels):
                var = ctk.BooleanVar(value=True)
                checkbox = ctk.CTkCheckBox(
                    int_channel_container, 
                    text=channel, 
                    variable=var,
                    font=ctk.CTkFont(size=12)
                )
                checkbox.pack(side="left", padx=8, pady=5)
                intensity_channels_vars.append(var)
            
            # Update scene information
            if scenes:
                scene_text = f"Available scenes: {', '.join(scenes[:5])}"
                if len(scenes) > 5:
                    scene_text += f" ... and {len(scenes)-5} more"
                scene_example_label.configure(text=scene_text, text_color="white")
            else:
                scene_example_label.configure(text="No scenes found", text_color="orange")
            
            update_status("✓ File metadata loaded", "green")
            
        except Exception as e:
            update_status(f"❌ Error reading file: {str(e)}", "red")
            messagebox.showerror("Error", f"Could not read file: {str(e)}")
    
    # Process Button Section - Always visible at bottom
    process_section = ctk.CTkFrame(main_scrollable, corner_radius=15, fg_color="transparent")
    process_section.grid(row=7, column=0, padx=20, pady=30, sticky="ew")
    process_section.grid_columnconfigure(0, weight=1)
    
    def check_ready_to_process():
        """Check if all required inputs are provided and enable/disable process button"""
        selected_files = [file for file, var in file_checkboxes if var.get()]
        has_files = len(selected_files) > 0
        has_savedir = bool(savedir_var.get())
        
        # Update button text based on selected options
        if get_tracking_var.get() and get_regionprops_var.get():
            action_text = "🚀 Start Segmentation, Tracking & Analysis"
        elif get_tracking_var.get():
            action_text = "🚀 Start Segmentation & Tracking"
        elif get_regionprops_var.get():
            action_text = "🚀 Start Segmentation & Analysis"
        else:
            action_text = "🚀 Start Segmentation Only"
        
        if has_files and has_savedir:
            process_button.configure(state="normal", text=action_text)
            ready_label.configure(text="✓ Ready to process", text_color="green")
        else:
            process_button.configure(state="disabled", text="⏳ Configure settings first")
            missing = []
            if not has_files:
                missing.append("files")
            if not has_savedir:
                missing.append("save directory")
            ready_label.configure(text=f"❌ Missing: {', '.join(missing)}", text_color="orange")
    
    ready_label = ctk.CTkLabel(
        process_section,
        text="❌ Missing: files, save directory",
        font=ctk.CTkFont(size=12),
        text_color="orange"
    )
    ready_label.grid(row=0, column=0, pady=(0, 10))
    
    def start_processing():
        # Validate inputs
        selected_files = [file for file, var in file_checkboxes if var.get()]
        if not selected_files:
            messagebox.showerror("Error", "No files selected!")
            return
        
        if not savedir_var.get():
            messagebox.showerror("Error", "No save directory selected!")
            return
        
        # Check tracking requirements
        if get_tracking_var.get():
            try:
                # Quick check if first file is time-lapse
                test_img = AICSImage(selected_files[0])
                if test_img.dims.T <= 1:
                    if not messagebox.askyesno("Tracking Warning", 
                        "Selected files appear to be single time-point images. "
                        "Tracking requires time-lapse data.\n\n"
                        "Continue with tracking disabled?"):
                        return
                    get_tracking_var.set(False)
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not verify time-lapse data: {e}")
        
        # Confirmation dialog
        file_count = len(selected_files)
        analysis_options = []
        if get_regionprops_var.get():
            analysis_options.append("region properties")
        if get_tracking_var.get():
            analysis_options.append("cell tracking")
        
        analysis_text = " and ".join(analysis_options) if analysis_options else "segmentation only"
        confirm_msg = f"Process {file_count} file{'s' if file_count > 1 else ''} with {analysis_text}?\n\nThis may take several minutes."
        
        if not messagebox.askyesno("Confirm Processing", confirm_msg):
            return
        
        try:
            update_status("🚀 Starting processing...", "blue")
            
            # Get selected channels
            segment_channel_index = int(selected_channel_var.get())
            intensity_channel_indices = [i for i, var in enumerate(intensity_channels_vars) if var.get()]
            
            # Setup model
            if custom_model_var.get() and model_path_var.get():
                model = models.CellposeModel(gpu=True, pretrained_model=model_path_var.get())
            else:
                if cellpose_version >= '4.0':
                    model = models.CellposeModel(gpu=True)
                else:
                    model = models.CellposeModel(gpu=True, pretrained_model='cyto3')
            
            # Get scene identifiers
            scene_ids = None
            if scene_identifiers_var.get().strip():
                scene_ids = [s.strip() for s in scene_identifiers_var.get().split(',')]
            
            # Close the app window
            app.destroy()
            
            # Choose processing function based on tracking option
            if get_tracking_var.get():
                process_files_multichannel_3(
                    selected_files, 
                    model, 
                    model_diameter_var.get(), 
                    savedir_var.get(),
                    get_regionprops=get_regionprops_var.get(),
                    get_tracking=True,
                    segment_channels=[segment_channel_index],
                    intensity_channels=intensity_channel_indices,
                    subset=slice(None),
                    chunksize=[1, 1],
                    scene_identifiers=scene_ids
                )
            else:
                process_files_multichannel_2(
                    selected_files, 
                    model, 
                    model_diameter_var.get(), 
                    savedir_var.get(),
                    get_regionprops=get_regionprops_var.get(),
                    segment_channels=[segment_channel_index],
                    intensity_channels=intensity_channel_indices,
                    subset=slice(None),
                    chunksize=[1, 1],
                    scene_identifiers=scene_ids
                )
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    process_button = ctk.CTkButton(
        process_section, 
        text="⏳ Configure settings first",
        command=start_processing,
        font=ctk.CTkFont(size=18, weight="bold"),
        height=50,
        width=400,
        state="disabled"
    )
    process_button.grid(row=1, column=0, pady=10)
    
    # Initialize the ready check
    check_ready_to_process()
    
    # Bind variables to ready check
    get_tracking_var.trace('w', lambda *args: check_ready_to_process())
    get_regionprops_var.trace('w', lambda *args: check_ready_to_process())
    savedir_var.trace('w', lambda *args: check_ready_to_process())
    
    app.mainloop()

if __name__ == "__main__":
    user_interface()

