from cellpose import io, models, train, version
from tkinter import filedialog as fd
from tkinter import Tk, messagebox
import os
import numpy as np
from tifffile import imread as tif_imread
import random
from nd2 import imread as nd2_imread
import time
from functools import wraps
from tqdm import tqdm
from utils import crop_large_image, stitch_images

io.logger_setup()

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def generate_training_test_sets(images, masks, proportion=0.7):
    """
    Applies the same random shuffle to two lists, then splits them into two
    equal-length subsets, retaining the original indices for each element.
    Useful for splitting data into training and testing sets.

    Args:
        images: The first list.
        masks: The second list.
        proportion: The proportion to include in the first subset.

    Returns:
        A tuple containing two tuples. Each inner tuple represents a subset pair:
        ((subset1_of_list1, subset1_of_list2), (subset2_of_list1, subset2_of_list2)).
        Each subset_of_listX is a list of (value, original_index) tuples.

    Raises:
        ValueError: If lists have different lengths.
    """
    n1 = len(images)
    n2 = len(masks)
    # Convert lists to numpy arrays to facilitate indexing
    images = np.array(images)
    masks = np.array(masks)

    if n1 != n2:
        raise ValueError("Uneven number of images and masks. Please check your data.")
    if n1 == 0 or n2 == 0:
        raise ValueError("Need at least one image and one mask")

    # 1. Create an index
    index = np.arange(n1)
    # 2. Shuffle the indices
    random.shuffle(index)
    # 3. Calculate the split point
    splitpoint = int(proportion*n1)

    # 4. Split the shuffled combined list into two halves
    shuffled_half1 = index[:splitpoint]
    shuffled_half2 = index[splitpoint:]

    # 5. Create the two subsets
    training_images = [images[i] for i in shuffled_half1]
    training_masks = [masks[i] for i in shuffled_half1]
    testing_images = [images[i] for i in shuffled_half2]
    testing_masks = [masks[i] for i in shuffled_half2]

    # training_masks = masks[shuffled_half1].tolist()
    # testing_images = images[shuffled_half2].tolist()
    # testing_masks = masks[shuffled_half2].tolist()
    

    return training_images, training_masks, testing_images, testing_masks

@timing
def train_cpSAM(images, labels, save_path, proportion=0.7, weight_decay=0.1, learning_rate=1e-5, n_epochs=100):
    """
    Trains a Cellpose SAM model on the provided data.

    Args:
        images: List of images to train on.
        labels: List of corresponding masks for the images.
        proportion: Proportion of training data in the dataset.
        weight_decay: Weight decay for the optimizer.
        learning_rate: Learning rate for the optimizer.
        n_epochs: Number of epochs to train the model.

    Returns:
        The path to the saved model and training losses.
    """
    # Split the data into training and testing sets
    train_images, train_labels, test_images, test_labels = generate_training_test_sets(images, labels, proportion=proportion)

    # Get save path for the model
    model_name = os.path.basename(save_path).split('.')[0]  # Get the name of the model from the save path
    save_dir = os.path.dirname(save_path)  # Get the directory to save the model in
    model = models.CellposeModel(gpu=True)
    print(f"Training Cellpose SAM model with {len(train_images)} training images and {len(test_images)} testing images.")
    print(f"Model will be saved to {save_dir} with name {model_name}.")
    print(f"Using proportion: {proportion}, weight decay: {weight_decay}, learning rate: {learning_rate}, epochs: {n_epochs}")
    # Train the model
    model_path, train_losses, test_losses = train.train_seg(model.net,
                                train_data=train_images, train_labels=train_labels,
                                test_data=test_images, test_labels=test_labels,
                                weight_decay=weight_decay, learning_rate=learning_rate,
                                n_epochs=n_epochs, model_name=model_name, save_path=save_dir)
    print(f"Model saved to {model_path}")
    return model_path, train_losses, test_losses


if __name__ == "__main__":
    # Make root window and make it hidden
    root = Tk()
    root.withdraw()  # Hide the root window
    root.call('wm', 'attributes', '.', '-topmost', True) # Keep the window on top
    confirmation = messagebox.askyesno("Cellpose SAM Training", "This script will train a Cellpose SAM model on your data. " \
    "\n You need a training data set with images and their corresponding masks." \
    " Image files should contain 'image' in the name, and mask files should contain 'masks' in the name." \
    " Continue? " )
    if not confirmation:
        print("Training cancelled by user.")
        root.destroy()
        exit()
    #Get user inputs
    data_dir = fd.askdirectory(title='Select directory with training data')
    save_path = fd.asksaveasfilename(title='Name and location of the new model')
    if not data_dir or not save_path:
        print("No directory or save path selected. Exiting.")
        root.destroy()
        exit()
    advanced = messagebox.askyesno("Advanced Options", "Do you want to set advanced options for training? (e.g. training data proportion, weight decay, learning rate, epochs)")
    if advanced:
        proportion = float(input("Enter proportion of training data in the dataset (default 0.7): ") or 0.7)
        weight_decay = float(input("Enter weight decay (default 0.1): ") or 0.1)
        learning_rate = float(input("Enter learning rate (default 1e-5): ") or 1e-5)
        n_epochs = int(input("Enter number of epochs (default 100): ") or 100)
    else:
        proportion = 0.7
        weight_decay = 0.1
        learning_rate = 1e-5
        n_epochs = 100
    
    root.destroy()  # Destroy the root window after getting inputs
    # Load data
    # Data comes from timelapses. So for each image in the training set, there is a corresponding mask, and both need to be split in lists
    files = os.listdir(data_dir)
    images = []
    labels = []
    for i, file in enumerate(tqdm(files, desc="Loading data", unit="file")):
        path = os.path.join(data_dir, file)
        if 'image' in file:
            #if file.endswith(".tif") or file.endswith(".tiff"):
            try:
                image = io.imread(path)
            #elif file.endswith(".npy"):
            except:
                if file.endswith(".npy"):
                    # If the file is a numpy array, load it
                    image = np.load(path, allow_pickle=True)
                elif file.endswith(".nd2"):
                    image = nd2_imread(path)
                elif file.endswith(".tiff") or file.endswith(".tif"):
                    image = tif_imread(path)
                else:
                    raise ValueError(f"Unsupported file format: {file}")
            images.append(image)
        elif 'masks' in file:
            if file.endswith(".tif") or file.endswith(".tiff"):
                mask = tif_imread(path)
            elif file.endswith(".npy"):
                mask = np.load(path)
            labels.append(mask)

    if len(images) == 1:
        images = crop_large_image(images[0], n_segments=2)  # Crop the image into smaller patches if needed
        labels = crop_large_image(labels[0], n_segments=2)  # Crop the mask into smaller patches if needed
    model_path, train_losses, test_losses = train_cpSAM(images, labels, save_path, proportion=proportion, weight_decay=weight_decay, learning_rate=learning_rate, n_epochs=n_epochs)
    
    # Plot training losses and testing losses vs epochs
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss vs Epochs')
    plt.legend()
    plt.show()

    print(f"Model saved to {model_path}")