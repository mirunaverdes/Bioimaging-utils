import numpy as np
import random
import os
from functools import wraps
from tifffile import imread as tif_imread
from nd2 import imread as nd2_imread
import time
from tqdm import tqdm   

class CropIterator:
    """Iterator for cropping large images into smaller patches using views with optional overlap."""
    
    def __init__(self, image, n_segments_x, n_segments_y, channel_axis=-1, overlap=0):
        self.image = image
        self.n_segments_x = n_segments_x
        self.n_segments_y = n_segments_y
        self.channel_axis = channel_axis
        self.overlap = overlap  # Overlap percentage (0-100)
        self.channels_moved = False
        
        # Handle channel axis repositioning
        if image.ndim > 2 and channel_axis != -1 and channel_axis != image.ndim - 1:
            self.image = np.moveaxis(image, channel_axis, -1)
            self.channels_moved = True
        
        self.height, self.width = self.image.shape[:2]
        
        # Calculate base segment dimensions
        base_segment_height = self.height // n_segments_y
        base_segment_width = self.width // n_segments_x
        
        # Calculate overlap in pixels
        self.overlap_height = int(base_segment_height * (overlap / 100))
        self.overlap_width = int(base_segment_width * (overlap / 100))
        
        # Actual segment dimensions including overlap
        self.segment_height = base_segment_height + self.overlap_height
        self.segment_width = base_segment_width + self.overlap_width
        
        # Step size between segments (reduced by overlap)
        self.step_height = base_segment_height
        self.step_width = base_segment_width
        
        self.total_segments = n_segments_x * n_segments_y
        
    def __iter__(self):
        self.current = 0
        return self
        
    def __next__(self):
        if self.current >= self.total_segments:
            raise StopIteration
            
        i = self.current // self.n_segments_x
        j = self.current % self.n_segments_x
        
        # Calculate starting positions with step size
        start_row = i * self.step_height
        start_col = j * self.step_width
        
        # Calculate end positions
        end_row = start_row + self.segment_height
        end_col = start_col + self.segment_width
        
        # Handle boundaries - ensure we don't go beyond image dimensions
        end_row = min(end_row, self.height)
        end_col = min(end_col, self.width)
        
        # For the last row/column, extend to include all remaining pixels
        if i == self.n_segments_y - 1:
            end_row = self.height
        if j == self.n_segments_x - 1:
            end_col = self.width
        
        # Create a view (not a copy)
        cropped_view = self.image[start_row:end_row, start_col:end_col]
        
        # Move channels back if they were moved
        if self.channels_moved:
            cropped_view = np.moveaxis(cropped_view, -1, self.channel_axis)
        
        self.current += 1
        return cropped_view
        
    def __len__(self):
        return self.total_segments
        
    def __getitem__(self, index):
        if index >= self.total_segments:
            raise IndexError("Index out of range")
            
        i = index // self.n_segments_x
        j = index % self.n_segments_x
        
        # Calculate starting positions with step size
        start_row = i * self.step_height
        start_col = j * self.step_width
        
        # Calculate end positions
        end_row = start_row + self.segment_height
        end_col = start_col + self.segment_width
        
        # Handle boundaries - ensure we don't go beyond image dimensions
        end_row = min(end_row, self.height)
        end_col = min(end_col, self.width)
        
        # For the last row/column, extend to include all remaining pixels
        if i == self.n_segments_y - 1:
            end_row = self.height
        if j == self.n_segments_x - 1:
            end_col = self.width
        
        cropped_view = self.image[start_row:end_row, start_col:end_col]
        
        if self.channels_moved:
            cropped_view = np.moveaxis(cropped_view, -1, self.channel_axis)
            
        return cropped_view
    
    def get_original_dimensions(self):
        """Return the original image dimensions."""
        return (self.height, self.width)
    
    def get_step_size(self):
        """Return the step size used for cropping."""
        return (self.step_height, self.step_width)


def crop_large_image(image, n_segments_x=1, n_segments_y=1, channel_axis=-1, overlap=0):
    """
    Crops a large image (NumPy or Dask array) into smaller patches using views.
    The last row and column will include any remaining pixels if dimensions don't divide evenly.
    
    Args:
        image (np.ndarray or dask.array): The input image to crop.
        n_segments_x (int): The number of segments to divide the image into (x-axis).
        n_segments_y (int): The number of segments to divide the image into (y-axis).
        channel_axis (int): The axis that represents the color channels (default: -1, i.e. last one).
        overlap (float): Overlap percentage between tiles (0-100). Default: 0 (no overlap).
    Returns:
        CropIterator: An iterator that yields cropped image patches as views (same type as input).
    """
    return CropIterator(image, n_segments_x, n_segments_y, channel_axis, overlap)

def stitch_images(images, n_segments_x=1, n_segments_y=1, channel_axis=-1):
    """
    Stitches a list of cropped images back into a single large image.
    Now handles variable tile sizes (for edge tiles that include remaining pixels).
    
    Args:
        images (list): A list of cropped image patches.
        n_segments_x (int): The number of segments the original image was divided into (x-axis).
        n_segments_y (int): The number of segments the original image was divided into (y-axis).
        channel_axis (int): The axis that represents the color channels (default: -1, i.e. last one).
    Returns:
        np.ndarray: The stitched large image.       
    """
    if not images:
        raise ValueError("No images provided for stitching")
    
    channels_moved = False
    if images[0].ndim > 2:
        if channel_axis != -1 and channel_axis != images[0].ndim - 1:
            images = [np.moveaxis(img, channel_axis, -1) for img in images]
            channels_moved = True

    # Calculate base segment dimensions from the first (top-left) tile
    base_segment_height = images[0].shape[0]
    base_segment_width = images[0].shape[1]
    
    # Calculate total dimensions by examining edge tiles
    total_height = (n_segments_y - 1) * base_segment_height + images[-n_segments_x].shape[0]  # Last row
    total_width = (n_segments_x - 1) * base_segment_width + images[n_segments_x - 1].shape[1]  # Last column
    
    # Initialize output image
    if len(images[0].shape) > 2:
        stitched_image = np.zeros((total_height, total_width, images[0].shape[2]), dtype=images[0].dtype)
    else:
        stitched_image = np.zeros((total_height, total_width), dtype=images[0].dtype)
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            index = i * n_segments_x + j
            if index >= len(images):
                continue
                
            current_image = images[index]
            
            # Calculate position
            start_row = i * base_segment_height
            start_col = j * base_segment_width
            end_row = start_row + current_image.shape[0]
            end_col = start_col + current_image.shape[1]
            
            # Place the image
            if current_image.ndim > 2:
                stitched_image[start_row:end_row, start_col:end_col, :] = current_image
            else:
                stitched_image[start_row:end_row, start_col:end_col] = current_image
    
    if channels_moved:
        stitched_image = np.moveaxis(stitched_image, -1, channel_axis)

    return stitched_image

def convert_to_minimal_format(masks:np.ndarray):
    """
    Converts the input arrays to a minimal format by reducing their data type. If the input is a list,
    it will be converted to a NumPy array first.
    Args:
        masks (np.ndarray/list of np.ndarray): The input arrays to convert.
    Returns:
        array (np.ndarray): Input array with minimal data type
    """
    # Convert masks to minimal data type
    if np.all(np.asarray(masks)< 255):
        # If the masks are not in uint8 format, convert them to uint8
        masks = np.asarray(masks).astype(np.uint8)
    elif np.all(np.asarray(masks)< 65535):
        # If the masks are not in uint16 format, convert them to uint16
        masks = np.asarray(masks).astype(np.uint16)
    elif np.all(np.asarray(masks)< 4294967295):
        # If the masks are not in uint32 format, convert them to uint32
        masks = np.asarray(masks).astype(np.uint32)
    else:
        # If the masks are not in a standard format, convert them to a standard format
        masks = np.asarray(masks)
    return masks

def namedir(savedir:str, base:str):
    """
    Generates a unique directory name for a specified base name in a given directory and then creates the new dir.
    Args:
        savedir (str): Root path for the new dir
        base (str): Name of the directory to be created
    """
    if not os.path.exists(savedir):
        raise ValueError(f"The specified directory {savedir} does not exist.")
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ','.']
    for char in illegal_chars:
        base = base.replace(char, '_')
    path = os.path.join(savedir,base)
    if os.path.exists(path):
        index=1
        base_new = base + f"_{index}"
        while os.path.exists(os.path.join(savedir,base_new)):
            index += 1
            base_new = base + f"_{index}"
        path = os.path.join(savedir,base_new)

    os.makedirs(path, exist_ok=True)
    return path


def namefile(savedir, base, prefix, suffix, ext):
    """
    Generates a unique filename for a specified directory, base name, prefix, and extension.
    Args:
        savedir (str): The directory where the file will be saved.
        base (str): The base name of the file.
        prefix (str): The prefix to append to the base name.
        suffix (str): The suffix to append to the base name.
        ext (str): The file extension (e.g., 'tif', 'png').
    Returns:
        str: The generated filepath with the specified format.
    """
    if not os.path.exists(savedir):
        raise ValueError(f"The specified directory {savedir} does not exist.")
    filename = os.path.join(f"{prefix}_{base}_{suffix}")

    # Remove any illegal characters from the filename
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ','.']
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    
    filename_full = filename + f".{ext}"

    if os.path.exists(os.path.join(savedir, filename_full)):
        index = 1
        filename_full = filename + f"_{index}.{ext}"
        while os.path.exists(os.path.join(savedir, filename_full)):
            index += 1
            filename_full = filename + f"_{index}.{ext}"
    filepath = os.path.join(savedir, filename_full)
    return filepath

def stitch_images_with_overlap(images, n_segments_x, n_segments_y, overlap=0, 
                              blend_method='average', channel_axis=-1, 
                              original_shape=None):
    """
    Stitches a list of overlapping cropped images back into a single large image.
    
    Args:
        images (list): A list of cropped image patches.
        n_segments_x (int): The number of segments the original image was divided into (x-axis).
        n_segments_y (int): The number of segments the original image was divided into (y-axis).
        overlap (float): Overlap percentage between tiles (0-100). Default: 0 (no overlap).
        blend_method (str): Method to handle overlaps ('average', 'crop', 'linear', 'min', 'max','overwrite'). Default: 'average'.
        channel_axis (int): The axis that represents the color channels (default: -1, i.e. last one).   
        original_shape (tuple, optional): Original image shape (height, width) to ensure exact reconstruction
    """
    if not images:
        raise ValueError("No images provided for stitching")
    
    # Handle channel axis repositioning
    channels_moved = False
    if images[0].ndim > 2 and channel_axis != -1 and channel_axis != images[0].ndim - 1:
        images = [np.moveaxis(img, channel_axis, -1) for img in images]
        channels_moved = True
    
    # Get dimensions from the first tile
    first_img = images[0]
    
    # Calculate output dimensions
    if original_shape is not None:
        output_height, output_width = original_shape
        # Calculate base dimensions for positioning
        base_height = output_height // n_segments_y
        base_width = output_width // n_segments_x
        
        # Calculate positions using the same logic as CropIterator
        row_positions = [i * base_height for i in range(n_segments_y)]
        col_positions = [j * base_width for j in range(n_segments_x)]
        
    else:
        # Calculate base dimensions (without overlap)
        if overlap > 0:
            # Back-calculate base dimensions from overlapped tile
            base_height = int(first_img.shape[0] / (1 + overlap / 100))
            base_width = int(first_img.shape[1] / (1 + overlap / 100))
        else:
            base_height = first_img.shape[0]
            base_width = first_img.shape[1]
        
        # Calculate total output dimensions by examining actual tile sizes
        max_row_height = [0] * n_segments_y
        max_col_width = [0] * n_segments_x
        
        # Find maximum dimensions for each row and column
        for i in range(n_segments_y):
            for j in range(n_segments_x):
                index = i * n_segments_x + j
                if index < len(images):
                    tile = images[index]
                    max_row_height[i] = max(max_row_height[i], tile.shape[0])
                    max_col_width[j] = max(max_col_width[j], tile.shape[1])
        
        # Calculate cumulative positions
        row_positions = [0]
        col_positions = [0]
        
        for i in range(n_segments_y - 1):
            if overlap > 0:
                step = int(max_row_height[i] / (1 + overlap / 100))
            else:
                step = max_row_height[i]
            row_positions.append(row_positions[-1] + step)
        
        for j in range(n_segments_x - 1):
            if overlap > 0:
                step = int(max_col_width[j] / (1 + overlap / 100))
            else:
                step = max_col_width[j]
            col_positions.append(col_positions[-1] + step)
        
        # Calculate total output size
        output_height = row_positions[-1] + max_row_height[-1]
        output_width = col_positions[-1] + max_col_width[-1]
    
    # Initialize output arrays
    if first_img.ndim > 2:
        output_shape = (output_height, output_width, first_img.shape[2])
    else:
        output_shape = (output_height, output_width)
    
    stitched_image = np.zeros(output_shape, dtype=np.float64)
    weight_map = np.zeros(output_shape[:2], dtype=np.float64)
    
    # Process each tile
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            index = i * n_segments_x + j
            if index >= len(images):
                continue
                
            current_tile = images[index].astype(np.float64)
            
            # Calculate tile position in output image
            start_row = row_positions[i] if original_shape is not None else row_positions[i]
            start_col = col_positions[j] if original_shape is not None else col_positions[j]
            end_row = min(start_row + current_tile.shape[0], output_height)
            end_col = min(start_col + current_tile.shape[1], output_width)
            
            # Calculate actual dimensions that will be placed
            actual_height = end_row - start_row
            actual_width = end_col - start_col
            
            # Crop tile to fit if necessary
            tile_to_place = current_tile[:actual_height, :actual_width]
            
            if blend_method == 'crop':
                # Calculate overlap regions to remove
                if overlap > 0:
                    overlap_height = int(base_height * (overlap / 100)) if original_shape is None else int(first_img.shape[0] * (overlap / 100) / (1 + overlap / 100))
                    overlap_width = int(base_width * (overlap / 100)) if original_shape is None else int(first_img.shape[1] * (overlap / 100) / (1 + overlap / 100))
                    
                    crop_top = overlap_height // 2 if i > 0 else 0
                    crop_bottom = overlap_height // 2 if i < n_segments_y - 1 else 0
                    crop_left = overlap_width // 2 if j > 0 else 0
                    crop_right = overlap_width // 2 if j < n_segments_x - 1 else 0
                    
                    # Ensure crops don't exceed tile dimensions
                    crop_bottom = min(crop_bottom, tile_to_place.shape[0] - crop_top - 1)
                    crop_right = min(crop_right, tile_to_place.shape[1] - crop_left - 1)
                    
                    tile_cropped = tile_to_place[crop_top:tile_to_place.shape[0]-crop_bottom,
                                               crop_left:tile_to_place.shape[1]-crop_right]
                    
                    new_start_row = start_row + crop_top
                    new_start_col = start_col + crop_left
                    new_end_row = new_start_row + tile_cropped.shape[0]
                    new_end_col = new_start_col + tile_cropped.shape[1]
                    
                    # Ensure we don't exceed output bounds
                    new_end_row = min(new_end_row, output_height)
                    new_end_col = min(new_end_col, output_width)
                    
                    # Adjust cropped tile if output bounds exceeded
                    final_height = new_end_row - new_start_row
                    final_width = new_end_col - new_start_col
                    tile_cropped = tile_cropped[:final_height, :final_width]
                    
                    if tile_cropped.ndim > 2:
                        stitched_image[new_start_row:new_end_row, new_start_col:new_end_col, :] = tile_cropped
                    else:
                        stitched_image[new_start_row:new_end_row, new_start_col:new_end_col] = tile_cropped
                else:
                    # No overlap, just place the tile
                    if tile_to_place.ndim > 2:
                        stitched_image[start_row:end_row, start_col:end_col, :] = tile_to_place
                    else:
                        stitched_image[start_row:end_row, start_col:end_col] = tile_to_place
                        
            elif blend_method == 'overwrite':
                # Simply overwrite (last tile wins)
                if tile_to_place.ndim > 2:
                    stitched_image[start_row:end_row, start_col:end_col, :] = tile_to_place
                else:
                    stitched_image[start_row:end_row, start_col:end_col] = tile_to_place
                    
            elif blend_method == 'linear':
                # Linear blending based on distance from tile center
                weights = _create_linear_weights(actual_height, actual_width)
                
                if tile_to_place.ndim > 2:
                    for c in range(tile_to_place.shape[2]):
                        stitched_image[start_row:end_row, start_col:end_col, c] += tile_to_place[:, :, c] * weights
                else:
                    stitched_image[start_row:end_row, start_col:end_col] += tile_to_place * weights
                
                weight_map[start_row:end_row, start_col:end_col] += weights
                
            else:
                # For average, max, min methods
                current_region = stitched_image[start_row:end_row, start_col:end_col]
                current_weights = weight_map[start_row:end_row, start_col:end_col]
                
                # Create mask for existing data
                existing_mask = current_weights > 0
                
                if blend_method == 'average':
                    if tile_to_place.ndim > 2:
                        for c in range(tile_to_place.shape[2]):
                            stitched_image[start_row:end_row, start_col:end_col, c] += tile_to_place[:, :, c]
                    else:
                        stitched_image[start_row:end_row, start_col:end_col] += tile_to_place
                    weight_map[start_row:end_row, start_col:end_col] += 1
                    
                elif blend_method == 'max':
                    if tile_to_place.ndim > 2:
                        for c in range(tile_to_place.shape[2]):
                            region = stitched_image[start_row:end_row, start_col:end_col, c]
                            stitched_image[start_row:end_row, start_col:end_col, c] = np.where(
                                existing_mask, np.maximum(region, tile_to_place[:, :, c]), tile_to_place[:, :, c])
                    else:
                        region = stitched_image[start_row:end_row, start_col:end_col]
                        stitched_image[start_row:end_row, start_col:end_col] = np.where(
                            existing_mask, np.maximum(region, tile_to_place), tile_to_place)
                    weight_map[start_row:end_row, start_col:end_col] = np.maximum(current_weights, 1)
                    
                elif blend_method == 'min':
                    if tile_to_place.ndim > 2:
                        for c in range(tile_to_place.shape[2]):
                            region = stitched_image[start_row:end_row, start_col:end_col, c]
                            stitched_image[start_row:end_row, start_col:end_col, c] = np.where(
                                existing_mask, np.minimum(region, tile_to_place[:, :, c]), tile_to_place[:, :, c])
                    else:
                        region = stitched_image[start_row:end_row, start_col:end_col]
                        stitched_image[start_row:end_row, start_col:end_col] = np.where(
                            existing_mask, np.minimum(region, tile_to_place), tile_to_place)
                    weight_map[start_row:end_row, start_col:end_col] = np.maximum(current_weights, 1)
    
    # Normalize by weights for average and linear blending
    if blend_method in ['average', 'linear']:
        # Avoid division by zero
        weight_map[weight_map == 0] = 1
        if stitched_image.ndim > 2:
            for c in range(stitched_image.shape[2]):
                stitched_image[:, :, c] /= weight_map
        else:
            stitched_image /= weight_map
    
    # Convert back to original dtype
    stitched_image = stitched_image.astype(first_img.dtype)
    
    # Move channels back if they were moved
    if channels_moved:
        stitched_image = np.moveaxis(stitched_image, -1, channel_axis)
    
    return stitched_image


def _create_linear_weights(height, width):
    """
    Create linear weights for blending, with maximum weight at the center.
    
    Args:
        height (int): Height of the tile
        width (int): Width of the tile
        
    Returns:
        np.ndarray: 2D array of weights
    """
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    center_y, center_x = height // 2, width // 2
    
    # Normalize distances to [0, 1]
    dist_y = np.abs(y - center_y) / (height / 2)
    dist_x = np.abs(x - center_x) / (width / 2)
    
    # Combine distances and invert (so center has highest weight)
    weights = 1 - np.maximum(dist_y, dist_x)
    weights = np.maximum(weights, 0.1)  # Minimum weight to avoid zeros
    
    return weights

def stitch_label_images_with_overlap(images, n_segments_x, n_segments_y, overlap=0, 
                                   blend_method='relabel_merge', channel_axis=-1):
    """
    WARNING: Experimental function, may not handle all edge cases.
    Stitches a list of overlapping label images back into a single large image.
    Special handling for discrete label values.
    
    Args:
        images (list): A list of cropped label image patches with potential overlap.
        n_segments_x (int): The number of segments the original image was divided into (x-axis).
        n_segments_y (int): The number of segments the original image was divided into (y-axis).
        overlap (float): Overlap percentage between tiles (0-100).
        blend_method (str): Method to handle overlapping regions. Options:
            - 'relabel_merge': Relabel all masks and merge based on overlap area
            - 'largest_area': Keep the label with the largest area in overlap region
            - 'priority_order': Later tiles have priority (overwrite earlier ones)
            - 'crop_overlap': Remove overlap regions entirely
            - 'vote': Majority vote in overlapping pixels
            - 'distance_priority': Priority based on distance from tile center
        channel_axis (int): The axis that represents the color channels (default: -1).
    
    Returns:
        np.ndarray: The stitched large label image.
    """
    if not images:
        raise ValueError("No images provided for stitching")
    
    # Handle channel axis repositioning
    channels_moved = False
    if images[0].ndim > 2 and channel_axis != -1 and channel_axis != images[0].ndim - 1:
        images = [np.moveaxis(img, channel_axis, -1) for img in images]
        channels_moved = True
    
    # Get dimensions
    first_img = images[0]
    tile_height, tile_width = first_img.shape[:2]
    
    # Calculate overlap in pixels
    base_height = int(tile_height / (1 + overlap / 100))
    base_width = int(tile_width / (1 + overlap / 100))
    overlap_height = int(base_height * (overlap / 100))
    overlap_width = int(base_width * (overlap / 100))
    
    # Calculate output image dimensions
    output_height = base_height * n_segments_y + overlap_height
    output_width = base_width * n_segments_x + overlap_width
    
    # Initialize output array
    if first_img.ndim > 2:
        output_shape = (output_height, output_width, first_img.shape[2])
    else:
        output_shape = (output_height, output_width)
    
    stitched_image = np.zeros(output_shape, dtype=first_img.dtype)
    
    if blend_method == 'crop_overlap':
        return _stitch_labels_crop_overlap(images, n_segments_x, n_segments_y, 
                                         overlap_height, overlap_width, stitched_image)
    
    elif blend_method == 'relabel_merge':
        return _stitch_labels_relabel_merge(images, n_segments_x, n_segments_y, 
                                          base_height, base_width, stitched_image)
    
    elif blend_method == 'largest_area':
        return _stitch_labels_largest_area(images, n_segments_x, n_segments_y, 
                                         base_height, base_width, tile_height, 
                                         tile_width, stitched_image)
    
    elif blend_method == 'priority_order':
        return _stitch_labels_priority_order(images, n_segments_x, n_segments_y, 
                                           base_height, base_width, tile_height, 
                                           tile_width, stitched_image)
    
    elif blend_method == 'vote':
        return _stitch_labels_vote(images, n_segments_x, n_segments_y, 
                                 base_height, base_width, tile_height, 
                                 tile_width, stitched_image)
    
    elif blend_method == 'distance_priority':
        return _stitch_labels_distance_priority(images, n_segments_x, n_segments_y, 
                                               base_height, base_width, tile_height, 
                                               tile_width, stitched_image)
    
    else:
        raise ValueError(f"Unknown blend method: {blend_method}")


def _stitch_labels_relabel_merge(images, n_segments_x, n_segments_y, base_height, 
                               base_width, stitched_image):
    """Relabel all masks globally and merge based on overlap area."""
    
    current_label = 1
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
                
            current_tile = images[tile_idx].copy()
            start_row = i * base_height
            start_col = j * base_width
            end_row = min(start_row + current_tile.shape[0], stitched_image.shape[0])
            end_col = min(start_col + current_tile.shape[1], stitched_image.shape[1])
            
            # Get unique labels in this tile
            unique_labels = np.unique(current_tile)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            
            # Relabel objects in this tile
            for orig_label in unique_labels:
                mask = current_tile == orig_label
                
                # Check for overlap with existing labels in the output image
                overlap_region = stitched_image[start_row:end_row, start_col:end_col]
                mask_region = mask[:end_row-start_row, :end_col-start_col]
                
                # Find overlapping labels in the existing stitched image
                overlapping_labels = np.unique(overlap_region[mask_region])
                overlapping_labels = overlapping_labels[overlapping_labels > 0]
                
                if len(overlapping_labels) > 0:
                    # Merge with existing label (choose the one with most overlap)
                    overlap_counts = [(label, np.sum((overlap_region == label) & mask_region)) 
                                    for label in overlapping_labels]
                    best_label = max(overlap_counts, key=lambda x: x[1])[0]
                    current_tile[mask] = best_label
                else:
                    # Assign new global label
                    current_tile[mask] = current_label
                    current_label += 1
            
            # Place the relabeled tile - use proper merging instead of maximum
            tile_region = current_tile[:end_row-start_row, :end_col-start_col]
            
            # Only place non-background pixels
            non_background_mask = tile_region > 0
            stitched_image[start_row:end_row, start_col:end_col][non_background_mask] = tile_region[non_background_mask]
    
    return stitched_image

def _stitch_labels_relabel_merge_advanced(images, n_segments_x, n_segments_y, base_height, 
                                        base_width, stitched_image):
    """Advanced relabeling that truly merges overlapping objects."""
    from skimage.measure import regionprops
    from collections import defaultdict
    
    # First pass: place all tiles with unique labels per tile
    current_label = 1
    tile_label_mapping = {}  # (tile_idx, orig_label) -> global_label
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
                
            current_tile = images[tile_idx].copy()
            start_row = i * base_height
            start_col = j * base_width
            end_row = min(start_row + current_tile.shape[0], stitched_image.shape[0])
            end_col = min(start_col + current_tile.shape[1], stitched_image.shape[1])
            
            # Relabel all objects in this tile with unique global labels
            unique_labels = np.unique(current_tile)
            unique_labels = unique_labels[unique_labels > 0]
            
            for orig_label in unique_labels:
                mask = current_tile == orig_label
                tile_label_mapping[(tile_idx, orig_label)] = current_label
                current_tile[mask] = current_label
                current_label += 1
            
            # Place tile in output
            tile_region = current_tile[:end_row-start_row, :end_col-start_col]
            non_bg_mask = tile_region > 0
            stitched_image[start_row:end_row, start_col:end_col][non_bg_mask] = tile_region[non_bg_mask]
    
    # Second pass: find overlapping objects and merge them
    overlap_threshold = 0.1  # 10% overlap to consider merging
    merge_groups = defaultdict(set)  # Groups of labels that should be merged
    
    # Analyze overlaps between adjacent tiles
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
            
            current_tile = images[tile_idx]
            start_row = i * base_height
            start_col = j * base_width
            
            # Check overlap with right neighbor
            if j < n_segments_x - 1:
                right_tile_idx = tile_idx + 1
                if right_tile_idx < len(images):
                    _find_and_group_overlaps(
                        images, tile_idx, right_tile_idx, 
                        tile_label_mapping, merge_groups, 
                        overlap_threshold, 'horizontal'
                    )
            
            # Check overlap with bottom neighbor
            if i < n_segments_y - 1:
                bottom_tile_idx = tile_idx + n_segments_x
                if bottom_tile_idx < len(images):
                    _find_and_group_overlaps(
                        images, tile_idx, bottom_tile_idx, 
                        tile_label_mapping, merge_groups, 
                        overlap_threshold, 'vertical'
                    )
    
    # Third pass: apply merges
    label_remapping = {}
    next_merged_label = 1
    
    for group in merge_groups.values():
        if len(group) > 1:
            # Choose the smallest label as the target
            target_label = min(group)
            for label in group:
                label_remapping[label] = target_label
    
    # Apply remapping to the stitched image
    for old_label, new_label in label_remapping.items():
        stitched_image[stitched_image == old_label] = new_label
    
    return stitched_image


def _find_and_group_overlaps(images, tile1_idx, tile2_idx, tile_label_mapping, 
                           merge_groups, threshold, direction):
    """Helper function to find overlapping objects between adjacent tiles."""
    # This is a simplified version - full implementation would calculate
    # actual overlap regions and IoU scores
    
    tile1 = images[tile1_idx]
    tile2 = images[tile2_idx]
    
    # Get overlap region (simplified)
    if direction == 'horizontal':
        # Right edge of tile1 overlaps with left edge of tile2
        overlap1 = tile1[:, -10:]  # Last 10 pixels
        overlap2 = tile2[:, :10]   # First 10 pixels
    else:
        # Bottom edge of tile1 overlaps with top edge of tile2
        overlap1 = tile1[-10:, :]
        overlap2 = tile2[:10, :]
    
    # Find unique labels in the overlap regions
    labels1 = np.unique(overlap1)
    labels2 = np.unique(overlap2)
    
    # Remove background label (assuming 0 is the background)
    labels1 = labels1[labels1 > 0]
    labels2 = labels2[labels2 > 0]
    
    for label1 in labels1:
        for label2 in labels2:
            # Check if these labels are already in the same group
            if label1 in merge_groups and label2 in merge_groups[label1]:
                continue
            
            # Get mask for each label in the overlap region
            mask1 = (overlap1 == label1)
            mask2 = (overlap2 == label2)
            
            # Calculate simple overlap metric (fraction of overlap region)
            overlap_metric = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
            
            if overlap_metric > threshold:
                # Merge these labels
                if label1 not in merge_groups:
                    merge_groups[label1] = set()
                merge_groups[label1].add(label2)
                
                if label2 not in merge_groups:
                    merge_groups[label2] = set()
                merge_groups[label2].add(label1)

def _stitch_labels_largest_area(images, n_segments_x, n_segments_y, base_height, 
                              base_width, tile_height, tile_width, stitched_image):
    """Keep the label with the largest total area in overlapping regions."""
    from skimage.measure import regionprops
    
    # First pass: collect all objects and their areas
    all_objects = {}  # label -> {'area': int, 'tile': int, 'coords': list}
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
                
            current_tile = images[tile_idx]
            props = regionprops(current_tile)
            
            for prop in props:
                global_coords = []
                for coord in prop.coords:
                    global_y = coord[0] + i * base_height
                    global_x = coord[1] + j * base_width
                    global_coords.append((global_y, global_x))
                
                key = (tile_idx, prop.label)
                all_objects[key] = {
                    'area': prop.area,
                    'coords': global_coords,
                    'label': prop.label
                }
    
    # Second pass: resolve overlaps by area
    placed_pixels = set()
    
    # Sort by area (largest first)
    sorted_objects = sorted(all_objects.items(), key=lambda x: x[1]['area'], reverse=True)
    
    for (tile_idx, orig_label), obj_info in sorted_objects:
        valid_coords = []
        for coord in obj_info['coords']:
            if coord not in placed_pixels and 0 <= coord[0] < stitched_image.shape[0] and 0 <= coord[1] < stitched_image.shape[1]:
                valid_coords.append(coord)
                placed_pixels.add(coord)
        
        # Place the object with a new label
        for coord in valid_coords:
            stitched_image[coord[0], coord[1]] = orig_label + tile_idx * 10000  # Ensure unique labels
    
    return stitched_image


def _stitch_labels_vote(images, n_segments_x, n_segments_y, base_height, base_width, 
                       tile_height, tile_width, stitched_image):
    """Use majority voting in overlapping regions."""
    vote_count = np.zeros((*stitched_image.shape, 10), dtype=np.int32)  # Max 10 overlapping tiles
    vote_labels = np.zeros((*stitched_image.shape, 10), dtype=stitched_image.dtype)
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
                
            current_tile = images[tile_idx]
            start_row = i * base_height
            start_col = j * base_width
            end_row = min(start_row + tile_height, stitched_image.shape[0])
            end_col = min(start_col + tile_width, stitched_image.shape[1])
            
            for y in range(start_row, end_row):
                for x in range(start_col, end_col):
                    tile_y = y - start_row
                    tile_x = x - start_col
                    if tile_y < current_tile.shape[0] and tile_x < current_tile.shape[1]:
                        label = current_tile[tile_y, tile_x]
                        if label > 0:  # Ignore background
                            # Find empty slot or existing label
                            for slot in range(10):
                                if vote_count[y, x, slot] == 0:
                                    vote_labels[y, x, slot] = label + tile_idx * 10000
                                    vote_count[y, x, slot] = 1
                                    break
                                elif vote_labels[y, x, slot] == label + tile_idx * 10000:
                                    vote_count[y, x, slot] += 1
                                    break
    
    # Resolve votes
    for y in range(stitched_image.shape[0]):
        for x in range(stitched_image.shape[1]):
            if np.any(vote_count[y, x] > 0):
                best_slot = np.argmax(vote_count[y, x])
                stitched_image[y, x] = vote_labels[y, x, best_slot]
    
    return stitched_image

def _stitch_labels_crop_overlap(images, n_segments_x, n_segments_y, overlap_height, 
                               overlap_width, stitched_image):
    """Remove overlap regions entirely from label images."""
    
    # Calculate base dimensions
    first_img = images[0]
    tile_height, tile_width = first_img.shape[:2]
    base_height = tile_height - overlap_height
    base_width = tile_width - overlap_width
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            index = i * n_segments_x + j
            if index >= len(images):
                continue
                
            current_tile = images[index]
            
            # Calculate crop regions to remove overlap
            crop_top = overlap_height // 2 if i > 0 else 0
            crop_bottom = overlap_height // 2 if i < n_segments_y - 1 else 0
            crop_left = overlap_width // 2 if j > 0 else 0
            crop_right = overlap_width // 2 if j < n_segments_x - 1 else 0
            
            # Ensure crops don't exceed tile dimensions
            crop_bottom = min(crop_bottom, current_tile.shape[0] - crop_top - 1) if crop_top > 0 else crop_bottom
            crop_right = min(crop_right, current_tile.shape[1] - crop_left - 1) if crop_left > 0 else crop_right
            
            # Crop the tile
            end_row = current_tile.shape[0] - crop_bottom if crop_bottom > 0 else current_tile.shape[0]
            end_col = current_tile.shape[1] - crop_right if crop_right > 0 else current_tile.shape[1]
            
            cropped_tile = current_tile[crop_top:end_row, crop_left:end_col]
            
            # Calculate position in output image
            start_row = i * base_height
            start_col = j * base_width
            out_end_row = start_row + cropped_tile.shape[0]
            out_end_col = start_col + cropped_tile.shape[1]
            
            # Ensure we don't exceed output bounds
            out_end_row = min(out_end_row, stitched_image.shape[0])
            out_end_col = min(out_end_col, stitched_image.shape[1])
            
            # Adjust cropped tile if output bounds exceeded
            final_height = out_end_row - start_row
            final_width = out_end_col - start_col
            cropped_tile = cropped_tile[:final_height, :final_width]
            
            # Place the cropped tile
            if cropped_tile.ndim > 2:
                stitched_image[start_row:out_end_row, start_col:out_end_col, :] = cropped_tile
            else:
                stitched_image[start_row:out_end_row, start_col:out_end_col] = cropped_tile
    
    return stitched_image


def _stitch_labels_priority_order(images, n_segments_x, n_segments_y, base_height, 
                                base_width, tile_height, tile_width, stitched_image):
    """Later tiles have priority and overwrite earlier ones."""
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
                
            current_tile = images[tile_idx]
            start_row = i * base_height
            start_col = j * base_width
            end_row = min(start_row + tile_height, stitched_image.shape[0])
            end_col = min(start_col + tile_width, stitched_image.shape[1])
            
            # Calculate actual dimensions to place
            actual_height = end_row - start_row
            actual_width = end_col - start_col
            
            # Crop current tile to fit the actual placement area
            tile_to_place = current_tile[:actual_height, :actual_width].copy()
            
            # Get unique labels in the cropped tile
            unique_labels = np.unique(tile_to_place)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            
            # Relabel objects in the cropped tile
            for orig_label in unique_labels:
                mask = tile_to_place == orig_label
                new_label = orig_label + tile_idx * 10000  # Ensure global uniqueness
                tile_to_place[mask] = new_label
            
            # Place the relabeled tile (overwrite existing labels)
            if tile_to_place.ndim > 2:
                # For multi-channel label images
                for c in range(tile_to_place.shape[2]):
                    mask = tile_to_place[:, :, c] > 0
                    stitched_image[start_row:end_row, start_col:end_col, c][mask] = tile_to_place[:, :, c][mask]
            else:
                mask = tile_to_place > 0
                stitched_image[start_row:end_row, start_col:end_col][mask] = tile_to_place[mask]
    
    return stitched_image


def _stitch_labels_distance_priority(images, n_segments_x, n_segments_y, base_height, 
                                   base_width, tile_height, tile_width, stitched_image):
    """Priority based on distance from tile center - closer to center wins."""
    
    # Create distance maps for each tile position
    distance_map = np.full(stitched_image.shape[:2], np.inf, dtype=np.float64)
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            tile_idx = i * n_segments_x + j
            if tile_idx >= len(images):
                continue
                
            current_tile = images[tile_idx]
            start_row = i * base_height
            start_col = j * base_width
            end_row = min(start_row + tile_height, stitched_image.shape[0])
            end_col = min(start_col + tile_width, stitched_image.shape[1])
            
            # Calculate actual dimensions
            actual_height = end_row - start_row
            actual_width = end_col - start_col
            
            # Create distance map for this tile (distance from center)
            tile_center_y = actual_height // 2
            tile_center_x = actual_width // 2
            
            y_coords, x_coords = np.ogrid[:actual_height, :actual_width]
            tile_distances = np.sqrt((y_coords - tile_center_y)**2 + (x_coords - tile_center_x)**2)
            
            # Relabel current tile
            relabeled_tile = current_tile[:actual_height, :actual_width].copy()
            unique_labels = np.unique(relabeled_tile)
            unique_labels = unique_labels[unique_labels > 0]
            
            for orig_label in unique_labels:
                mask = relabeled_tile == orig_label
                new_label = orig_label + tile_idx * 10000
                relabeled_tile[mask] = new_label
                
                # Only place pixels that are closer to this tile's center
                placement_mask = mask & (tile_distances < distance_map[start_row:end_row, start_col:end_col])
                
                if relabeled_tile.ndim > 2:
                    for c in range(relabeled_tile.shape[2]):
                        stitched_image[start_row:end_row, start_col:end_col, c][placement_mask] = new_label
                else:
                    stitched_image[start_row:end_row, start_col:end_col][placement_mask] = new_label
                
                # Update distance map
                distance_map[start_row:end_row, start_col:end_col][placement_mask] = tile_distances[placement_mask]
    
    return stitched_image

def test_crop_stitch_functions(image_path=None, n_segments_x=2, n_segments_y=2, overlap=10):
    """
    Test function for cropping and stitching functions with matplotlib visualization.
    
    Args:
        image_path (str): Path to test image. If None, opens file dialog.
        n_segments_x (int): Number of segments in x direction
        n_segments_y (int): Number of segments in y direction
        overlap (float): Overlap percentage for testing
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from aicsimageio import AICSImage
    from tkinter.filedialog import askopenfilename
    from tkinter import Tk
    import numpy as np
    
    # Get image path if not provided
    if image_path is None:
        root = Tk()
        root.withdraw()
        image_path = askopenfilename(
            title="Select test image", 
            filetypes=[("Image files", "*.tif *.tiff *.nd2 *.png *.jpg *.jpeg")]
        )
        root.destroy()
        
        if not image_path:
            print("No image selected")
            return
    
    print(f"Testing with image: {image_path}")
    print(f"Segments: {n_segments_x}x{n_segments_y}, Overlap: {overlap}%")
    
    # Load image
    try:
        if image_path.endswith(('.tif', '.tiff')):
            image_data = tif_imread(image_path)
        elif image_path.endswith('.nd2'):
            image_data = nd2_imread(image_path)
        else:
            # Try with AICSImage for other formats
            img = AICSImage(image_path)
            image_data = img.get_image_data("YX")
        
        print(f"Raw image shape: {image_data.shape}")
        
        # Handle different image dimensions
        if image_data.ndim > 2:
            # For multi-channel or time series images
            if image_data.ndim == 3:
                if image_data.shape[0] <= 10:  # Likely channels or time points
                    # Take the first channel/timepoint
                    image_data = image_data[0]
                    print(f"Selected first channel/timepoint, new shape: {image_data.shape}")
                else:
                    # Assume it's (Y, X, C) format
                    if image_data.shape[2] <= 10:  # Likely channels
                        image_data = image_data[:, :, 0]
                        print(f"Selected first channel from (Y, X, C), new shape: {image_data.shape}")
                    else:
                        # Take middle slice if it's a Z-stack
                        image_data = image_data[image_data.shape[0] // 2]
                        print(f"Selected middle slice, new shape: {image_data.shape}")
            elif image_data.ndim == 4:
                # Likely (T, C, Y, X) or (T, Y, X, C)
                image_data = image_data[0, 0] if image_data.shape[1] < image_data.shape[2] else image_data[0, :, :, 0]
                print(f"Selected first timepoint and channel, new shape: {image_data.shape}")
            elif image_data.ndim > 4:
                # Just take the first slice of each dimension until we get 2D
                while image_data.ndim > 2:
                    image_data = image_data[0]
                print(f"Reduced to 2D, new shape: {image_data.shape}")
        
        # Ensure we have a 2D array
        if image_data.ndim != 2:
            raise ValueError(f"Could not reduce image to 2D. Final shape: {image_data.shape}")
        
        # Normalize to 0-1 for display
        if image_data.dtype != np.float64:
            image_data = image_data.astype(np.float64)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create synthetic test data
        print("Creating synthetic test image...")
        image_data = _create_synthetic_test_image()
    
    print(f"Final image shape for testing: {image_data.shape}")
    
    # Test 1: No overlap cropping and stitching
    print("\n1. Testing no-overlap cropping and stitching...")
    crops_no_overlap = crop_large_image(image_data, n_segments_x, n_segments_y, overlap=0)
    crops_no_overlap_list = list(crops_no_overlap)
    stitched_no_overlap = stitch_images(crops_no_overlap_list, n_segments_x, n_segments_y)
    
    # Test 2: Overlap cropping and stitching with different methods
    print(f"2. Testing {overlap}% overlap cropping and stitching...")
    crops_overlap = crop_large_image(image_data, n_segments_x, n_segments_y, overlap=overlap)
    crops_overlap_list = list(crops_overlap)
    
    # Test different blending methods
    blend_methods = ['average', 'max', 'crop', 'linear']
    stitched_results = {}
    
    for method in blend_methods:
        print(f"   - Testing {method} blending...")
        stitched_results[method] = stitch_images_with_overlap(
            crops_overlap_list, n_segments_x, n_segments_y, 
            overlap=overlap, blend_method=method,
            original_shape=crops_overlap.get_original_dimensions()
        )
    
    # Test 3: Label image stitching (create synthetic labels)
    print("3. Testing label image stitching...")
    synthetic_labels = _create_synthetic_labels(image_data.shape, n_segments_x, n_segments_y)
    label_crops = crop_large_image(synthetic_labels, n_segments_x, n_segments_y, overlap=overlap)
    label_crops_list = list(label_crops)
    
    label_methods = ['crop_overlap', 'relabel_merge', 'priority_order']
    label_results = {}
    
    for method in label_methods:
        print(f"   - Testing {method} label blending...")
        try:
            if method == 'relabel_merge':
                # Check if scikit-image is available
                try:
                    from skimage.measure import regionprops
                    label_results[method] = stitch_label_images_with_overlap(
                        label_crops_list, n_segments_x, n_segments_y, 
                        overlap=overlap, blend_method=method
                    )
                except ImportError:
                    print(f"   - Skipping {method}: scikit-image not available")
                    label_results[method] = None
            else:
                label_results[method] = stitch_label_images_with_overlap(
                    label_crops_list, n_segments_x, n_segments_y, 
                    overlap=overlap, blend_method=method
                )
        except Exception as e:
            print(f"   - Error with {method}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            label_results[method] = None
    
    # Create visualization
    _visualize_test_results(
        image_data, crops_no_overlap_list, crops_overlap_list,
        stitched_no_overlap, stitched_results, 
        synthetic_labels, label_results,
        n_segments_x, n_segments_y, overlap
    )


def _create_synthetic_test_image(size=(512, 512)):
    """Create a synthetic test image with patterns."""
    image = np.zeros(size)
    
    # Add gradients
    y, x = np.ogrid[:size[0], :size[1]]
    image += (x / size[1]) * 0.3
    image += (y / size[0]) * 0.2
    
    # Add circular patterns
    for i in range(3):
        for j in range(3):
            center_y = (i + 1) * size[0] // 4
            center_x = (j + 1) * size[1] // 4
            radius = 30
            
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            circle = np.exp(-((dist - radius) / 10)**2)
            image += circle * 0.5
    
    # Add some noise
    image += np.random.random(size) * 0.1
    
    return image


def _create_synthetic_labels(shape, n_segments_x, n_segments_y):
    """Create synthetic label image for testing."""
    labels = np.zeros(shape, dtype=np.uint16)
    
    label_id = 1
    segment_height = shape[0] // (n_segments_y * 2)
    segment_width = shape[1] // (n_segments_x * 2)
    
    for i in range(n_segments_y * 2):
        for j in range(n_segments_x * 2):
            start_y = i * segment_height + 10
            end_y = (i + 1) * segment_height - 10
            start_x = j * segment_width + 10
            end_x = (j + 1) * segment_width - 10
            
            if start_y < shape[0] and start_x < shape[1]:
                end_y = min(end_y, shape[0])
                end_x = min(end_x, shape[1])
                labels[start_y:end_y, start_x:end_x] = label_id
                label_id += 1
    
    return labels


def _visualize_test_results(original, crops_no_overlap, crops_overlap,
                          stitched_no_overlap, stitched_results, 
                          synthetic_labels, label_results,
                          n_segments_x, n_segments_y, overlap):
    """Create comprehensive visualization of test results."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Calculate figure size and layout
    n_intensity_methods = len(stitched_results)
    n_label_methods = len([r for r in label_results.values() if r is not None])
    
    fig = plt.figure(figsize=(20, 16))
    
    # Row 1: Original and crops visualization
    ax1 = plt.subplot(4, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    _add_grid_overlay(ax1, original.shape, n_segments_x, n_segments_y, overlap)
    plt.axis('off')
    
    # Show individual crops
    ax2 = plt.subplot(4, 4, 2)
    crop_montage = _create_crop_montage(crops_no_overlap, n_segments_x, n_segments_y)
    plt.imshow(crop_montage, cmap='gray')
    plt.title('No Overlap Crops')
    plt.axis('off')
    
    ax3 = plt.subplot(4, 4, 3)
    crop_montage_overlap = _create_crop_montage(crops_overlap, n_segments_x, n_segments_y)
    plt.imshow(crop_montage_overlap, cmap='gray')
    plt.title(f'{overlap}% Overlap Crops')
    plt.axis('off')
    
    # Row 2: No overlap stitching result and difference
    ax4 = plt.subplot(4, 4, 5)
    plt.imshow(stitched_no_overlap, cmap='gray')
    plt.title('Stitched (No Overlap)')
    plt.axis('off')
    
    ax5 = plt.subplot(4, 4, 6)
    diff_no_overlap = np.abs(original - stitched_no_overlap)
    plt.imshow(diff_no_overlap, cmap='hot')
    plt.title('Difference (No Overlap)')
    plt.colorbar(shrink=0.6)
    plt.axis('off')
    
    # Row 3: Overlap stitching results
    plot_idx = 7
    for method, result in stitched_results.items():
        if plot_idx > 10:
            break
        ax = plt.subplot(4, 4, plot_idx)
        plt.imshow(result, cmap='gray')
        plt.title(f'Stitched ({method})')
        plt.axis('off')
        plot_idx += 1
    
    # Row 4: Label stitching results
    ax_label_orig = plt.subplot(4, 4, 13)
    plt.imshow(synthetic_labels, cmap='tab20')
    plt.title('Synthetic Labels')
    plt.axis('off')
    
    plot_idx = 14
    for method, result in label_results.items():
        if result is not None and plot_idx <= 16:
            ax = plt.subplot(4, 4, plot_idx)
            plt.imshow(result, cmap='tab20')
            plt.title(f'Labels ({method})')
            plt.axis('off')
            plot_idx += 1
    
    # Add metrics text
    _add_metrics_text(fig, original, stitched_no_overlap, stitched_results)
    
    plt.tight_layout()
    plt.suptitle(f'Crop/Stitch Test Results - {n_segments_x}x{n_segments_y} segments, {overlap}% overlap', 
                 fontsize=16, y=0.98)
    plt.show()
    
    # Print summary statistics
    _print_summary_stats(original, stitched_no_overlap, stitched_results, label_results)


def _add_grid_overlay(ax, shape, n_segments_x, n_segments_y, overlap):
    """Add grid overlay showing crop boundaries."""
    from matplotlib.patches import Rectangle
    height, width = shape
    segment_height = height // n_segments_y
    segment_width = width // n_segments_x
    
    # Calculate overlap
    overlap_height = int(segment_height * (overlap / 100))
    overlap_width = int(segment_width * (overlap / 100))
    
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            start_row = i * segment_height
            start_col = j * segment_width
            
            # Base rectangle
            rect = Rectangle((start_col, start_row), segment_width, segment_height,
                           linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Overlap regions if applicable
            if overlap > 0:
                # Add overlap visualization
                overlap_rect = Rectangle(
                    (start_col, start_row), 
                    segment_width + overlap_width, 
                    segment_height + overlap_height,
                    linewidth=1, edgecolor='blue', facecolor='none', linestyle='--'
                )
                ax.add_patch(overlap_rect)


def _create_crop_montage(crops, n_segments_x, n_segments_y):
    """Create a montage of all crops for visualization."""
    crop_list = list(crops) if hasattr(crops, '__iter__') else crops
    
    if not crop_list:
        return np.zeros((100, 100))
    
    # Get maximum dimensions
    max_h = max(crop.shape[0] for crop in crop_list)
    max_w = max(crop.shape[1] for crop in crop_list)
    
    # Create montage
    montage = np.zeros((max_h * n_segments_y, max_w * n_segments_x))
    
    for i, crop in enumerate(crop_list):
        row = i // n_segments_x
        col = i % n_segments_x
        
        if row < n_segments_y:
            start_row = row * max_h
            start_col = col * max_w
            end_row = start_row + crop.shape[0]
            end_col = start_col + crop.shape[1]
            
            montage[start_row:end_row, start_col:end_col] = crop
    
    return montage


def _add_metrics_text(fig, original, stitched_no_overlap, stitched_results):
    """Add metrics text to the figure."""
    metrics_text = "Reconstruction Metrics:\n"
    
    # Helper function to safely calculate metrics with shape checking
    def safe_calculate_metrics(img1, img2, method_name):
        if img1.shape != img2.shape:
            # Crop to minimum common size
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1_cropped = img1[:min_h, :min_w]
            img2_cropped = img2[:min_h, :min_w]
            metrics_text_local = f"{method_name} (cropped {img1.shape}->{img1_cropped.shape})"
        else:
            img1_cropped = img1
            img2_cropped = img2
            metrics_text_local = method_name
            
        mse = np.mean((img1_cropped - img2_cropped)**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        return mse, psnr, metrics_text_local
    
    # No overlap metrics
    mse, psnr, method_text = safe_calculate_metrics(original, stitched_no_overlap, "No Overlap")
    metrics_text += f"{method_text} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB\n"
    
    # Overlap methods metrics
    for method, result in stitched_results.items():
        mse, psnr, method_text = safe_calculate_metrics(original, result, method.capitalize())
        metrics_text += f"{method_text} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB\n"
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _print_summary_stats(original, stitched_no_overlap, stitched_results, label_results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Original image shape: {original.shape}")
    print(f"Original image range: [{original.min():.3f}, {original.max():.3f}]")
    
    print(f"\nNo overlap reconstruction:")
    print(f"  Stitched shape: {stitched_no_overlap.shape}")
    print(f"  Shape match: {original.shape == stitched_no_overlap.shape}")
    
    # Handle shape mismatch for MSE calculation
    if original.shape != stitched_no_overlap.shape:
        min_h = min(original.shape[0], stitched_no_overlap.shape[0])
        min_w = min(original.shape[1], stitched_no_overlap.shape[1])
        orig_crop = original[:min_h, :min_w]
        stitch_crop = stitched_no_overlap[:min_h, :min_w]
        mse = np.mean((orig_crop - stitch_crop)**2)
        print(f"  MSE (cropped to {orig_crop.shape}): {mse:.8f}")
        print(f"  Perfect reconstruction: False (shape mismatch)")
    else:
        mse = np.mean((original - stitched_no_overlap)**2)
        print(f"  MSE: {mse:.8f}")
        print(f"  Perfect reconstruction: {np.allclose(original, stitched_no_overlap, atol=1e-10)}")
    
    print(f"\nOverlap reconstruction methods:")
    for method, result in stitched_results.items():
        print(f"  {method} shape: {result.shape}")
        if original.shape != result.shape:
            min_h = min(original.shape[0], result.shape[0])
            min_w = min(original.shape[1], result.shape[1])
            orig_crop = original[:min_h, :min_w]
            result_crop = result[:min_h, :min_w]
            mse = np.mean((orig_crop - result_crop)**2)
            print(f"  {method} MSE (cropped): {mse:.8f}")
        else:
            mse = np.mean((original - result)**2)
            print(f"  {method} MSE: {mse:.8f}")
    
    print(f"\nLabel reconstruction methods:")
    for method, result in label_results.items():
        if result is not None:
            n_labels = len(np.unique(result)) - 1  # Exclude background
            print(f"  {method}: {n_labels} unique labels, shape: {result.shape}")
        else:
            print(f"  {method}: Failed")

def find_channel_axis(img):
    """
    Find the channel axis in an image array.
    
    Args:
        img (np.ndarray): Image array of shape (C, Y, X), (Y, C, X), or (X, Y, C)
        
    Returns:
        int: Index of the channel axis
    """
    if img.ndim == 3:
        # Find the axis with the smallest size
        channel_axis = np.argmin(img.shape)
        return channel_axis
    else:
        raise ValueError("Image must be 3D or 4D array")

if __name__ == "__main__":
    # Run the original tests
    from aicsimageio import AICSImage
    import numpy as np
    from tkinter.filedialog import askopenfilename
    from napari import Viewer
    from tkinter import Tk
    path = "C:\\Users\\s03434mv\\The University of Manchester Dropbox\\Miruna Verdes\\Bioimaging\\Code\\DasHUND\\Data for tests\\Test Inputs\\sox10_single_well-1_short.tif"
    print("Running comprehensive crop/stitch tests...")
    
    # Test with different configurations
    test_configurations = [
        {'image_path': path, 'n_segments_x': 2, 'n_segments_y': 2, 'overlap': 0},
        {'image_path': path, 'n_segments_x': 2, 'n_segments_y': 2, 'overlap': 10},
        {'image_path': path, 'n_segments_x': 3, 'n_segments_y': 3, 'overlap': 15},
    ]
    
    for config in test_configurations:
        print(f"\nTesting configuration: {config}")
        test_crop_stitch_functions(**config)