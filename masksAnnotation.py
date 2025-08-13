import numpy as np
from magicgui import magicgui
from skimage import measure, morphology
from tkinter import filedialog, messagebox, simpledialog
from skimage import measure
from skimage.measure import regionprops_table 
from pandas import DataFrame
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn.functional as F

PROPERTIES = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std')


def load_mask(mask_path: str):
    """
    Load a mask from a file.
    
    Args:
        mask_path (str): Path to the mask file.
        
    Returns:
        np.ndarray: Loaded mask.
    """
    return np.load(mask_path)

def save_mask(mask: np.ndarray, save_path: str):
    """
    Save a mask to a file.
    
    Args:
        mask (np.ndarray): Mask to save.
        save_path (str): Path to the save location.
    """
    np.save(save_path, mask)

def binary_erosion(mask, thickness):
    """
    Perform binary erosion using PyTorch convolution.
    Args:
        mask (torch.Tensor): Binary mask (0/1), shape (H, W), on GPU.
        thickness (int): Number of erosion iterations.
    Returns:
        torch.Tensor: Eroded mask (0/1), shape (H, W), on GPU.
    """
    device = mask.device
    kernel = torch.ones((1, 1, 3, 3), device=device)
    mask = mask.unsqueeze(0).unsqueeze(0).float()  # shape (1, 1, H, W)
    for _ in range(thickness):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask == kernel.numel()).float()
    return mask.squeeze(0).squeeze(0)

def extract_membrane(region_mask, thickness):
    """
    Extract membrane and cytoplasm from a binary region mask using PyTorch.
    Args:
        region_mask (torch.Tensor): Binary mask (0/1), shape (H, W), on GPU.
        thickness (int): Thickness of the membrane.
    Returns:
        membrane (torch.Tensor): Membrane mask (0/1), shape (H, W), on GPU.
        cytoplasm (torch.Tensor): Cytoplasm mask (0/1), shape (H, W), on GPU.
    """
    eroded = binary_erosion(region_mask, thickness)
    cytoplasm = eroded
    membrane = region_mask * (1 - eroded)
    return membrane, cytoplasm

def process_label(args):
    label, labels, thickness = args
    region_mask = labels == label
    membrane, cytoplasm = extract_membrane(region_mask, thickness)
    return label, membrane, cytoplasm

def extract_membrane_from_labels(labels: np.ndarray, thickness: int = 5):
    """
    Extract membrane from labeled regions.

    Args:
        labels (np.ndarray): Labeled regions mask.
        thickness (int): Thickness of the membrane to extract in pixels

    Returns:
        np.ndarray: Extracted labels of membranes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels_torch = torch.from_numpy(labels).to(device).int()  # <-- fix here
    membrane_mask = torch.zeros_like(labels_torch)
    cytoplasm_mask = torch.zeros_like(labels_torch)
    unique_labels = torch.unique(labels_torch)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        region_mask = (labels_torch == label).float()
        membrane, cytoplasm = extract_membrane(region_mask, thickness)
        membrane_mask[membrane > 0] = label
        cytoplasm_mask[cytoplasm > 0] = label

    return membrane_mask.cpu().numpy(), cytoplasm_mask.cpu().numpy()

if __name__=="__main__":
    # @magicgui(
    #     mask_path={"label": "Mask File Path"},
    #     save_path={"label": "Save Path"},
    #     thickness={"label": "Membrane Thickness (px)", "min": 1, "max": 20, "step": 1, "value": 5},
    #     call_button="Extract Membrane"
    # )
    def extract_membrane_widget(mask_path: str = "", save_path: str = "", thickness: int = 5):
        mask = load_mask(mask_path)
        membrane, cytoplasm = extract_membrane_from_labels(mask, thickness)
        save_mask(membrane, save_path + "_membrane.npy")
        save_mask(cytoplasm, save_path + "_cytoplasm.npy")
        print(f"Membrane and cytoplasm masks saved to {save_path}_membrane.npy and {save_path}_cytoplasm.npy")
    
    def extract_membrane_with_regionprops(image_path: str = "", mask_path: str = "", save_path: str = "", thickness: int = 5):
        mask = np.load(mask_path)
        image = np.load(image_path)  # Assuming image is a numpy array
        membrane, cytoplasm = extract_membrane_from_labels(mask, thickness)
        image = np.moveaxis(image, 0, -1)  # Move the channel axis to the last position
        props = regionprops_table(membrane, image, properties=PROPERTIES)
        df = DataFrame(props)
        file = mask_path.split("/")[-1].split(".")[0] + "_membrane"
        df['file'] = file  # or 'z' for z-stack
        df.to_csv(os.path.join(save_path, f"{file}_regionprops.csv"), index=False)
        save_mask(membrane, save_path + "_membrane.npy")
        save_mask(cytoplasm, save_path + "_cytoplasm.npy")
        print(f"Membrane and cytoplasm masks saved to {save_path}_membrane.npy and {save_path}_cytoplasm.npy")
    # # Register the widget
    # from napari_tools_menu import register_dock_widget
    # register_dock_widget(extract_membrane_widget, name="Extract Membrane")
    # # Run the widget
    # extract_membrane_widget.show(run=True)  # Show the widget

    image_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=[("NumPy files", "*.npy")])
    if not image_paths:
        messagebox.showerror("Error", "No image file selected.")
        quit()
    mask_paths = filedialog.askopenfilenames(title="Select Mask Files", filetypes=[("NumPy files", "*.npy")])
    if not mask_paths:
        messagebox.showerror("Error", "No mask file selected.")
        quit()

    save_path = filedialog.askdirectory(title="Select Save Location")
    if not save_path:
        messagebox.showerror("Error", "No save location selected.")
        quit()


    thickness = simpledialog.askinteger("Input", "Enter membrane thickness (px):", minvalue=1, maxvalue=20, initialvalue=5)
    if thickness is None:
        thickness = 5

    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Processing images"):
        extract_membrane_with_regionprops(image_path, mask_path, save_path, thickness)
