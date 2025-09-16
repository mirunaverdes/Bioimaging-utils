import numpy as np
import pandas as pd
import networkx as nx
from skimage import measure
import pickle
from tkinter import filedialog, messagebox, Tk, simpledialog
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
from tkinter import Tk, filedialog as fd
import os
from tifffile import imread
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_for_tracking():
    """ Function to load images and corresponding masks.

    Parameters:
    identifiers (list of strings): identifiers are unique strings that link images and corresponding masks.

    Returns:
    images (nested list of numpy arrays): list of images. 
    masks (nested list of numpy arrays): list of masks corresponding to the images.
    identifiers (list of strings): list of identifiers linking images and masks.
    
    Note:
    The function uses a GUI to select a directory containing images and masks.
    images and masks should be in the same directory, and the filenames should follow the pattern:
    "image_<identifier>.npy" for images and "masks_<anything>_<identifier>.npy" for masks.

    The function will load all images and masks in the selected directory that match the naming pattern.
    The images and masks are timelapses, loaded as lists of numpy arrays.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    dirpath = fd.askdirectory("Select directory with images and masks")
    files = os.listdir(dirpath)
    masks = []
    images = []
    identifiers = []
    # Loop through the files in the directory
    for file in files:
        if os.path.isfile(os.path.join(dirpath, file)):
            if "masks" in file and file.endswith(".npy"):
                # Get the identifier from the filename
                identifier = file.split("_")[-1].split(".npy")[0]
                # Find the corresponding image file
                image_file = f"image_{identifier}.npy"
                if image_file in files:
                    # Load the image and mask
                    images.append(np.load(os.path.join(dirpath, image_file), allow_pickle=True).tolist())
                    masks.append(np.load(os.path.join(dirpath, file), allow_pickle=True).tolist())
                    identifiers.append(identifier)
    root.destroy()  # Destroy the root window after getting inputs
    return images, masks, identifiers   

def preprocess_masks(masks):
    """ Function to preprocess masks by adding a mask in the frames with no masks detected.

    Parameters:
    masks (list of numpy arrays): list of masks.

    Returns:
    masks_altered (list of numpy array): preprocessed masks with frames containing no masks altered.
    """
    masks_altered = masks.copy()
    
    for i, timeframe in enumerate(masks):
        if len(np.unique(timeframe)) <= 1:
            # If no masks detected, create a mask 10 pixels wide, 1 pixel high in the top left corner
            masks_altered[i][0:10, 0:1] = 100

    return masks_altered

def track_Trackastra(images, masks):
    """ Function to track cells in images using Trackastra.

    Parameters:
    images (list of numpy arrays or numpy array): list of images or 3D array (T, Y, X).
    masks (list of numpy arrays or numpy array): list of masks corresponding to the images or 3D array (T, Y, X).

    Returns:
    track_graph (Trackastra graph): graph containing the tracked cells.
    ctc_tracks (pd.DataFrame): DataFrame containing the cell tracking data.
    masks_tracked (np.ndarray): Tracked masks of the cells.
    napari_tracks (napari.types.Tracks): Napari tracks of the cells.
    """
    
    # Convert inputs to numpy arrays if they're lists
    if isinstance(images, list):
        # Convert list to numpy array
        images_array = np.array(images)
    else:
        # Already a numpy array
        images_array = images
    
    if isinstance(masks, list):
        # Convert list to numpy array
        masks_array = np.array(masks)
    else:
        # Already a numpy array
        masks_array = masks
    
    # Ensure arrays are 3D (T, Y, X)
    if images_array.ndim == 2:
        images_array = images_array[np.newaxis, ...]  # Add time dimension
    if masks_array.ndim == 2:
        masks_array = masks_array[np.newaxis, ...]  # Add time dimension
    
    # Validate dimensions
    if images_array.shape != masks_array.shape:
        raise ValueError(f"Images and masks must have the same shape. "
                        f"Got images: {images_array.shape}, masks: {masks_array.shape}")
    
    # Ensure correct data types
    images_array = images_array.astype(np.float32)
    masks_array = masks_array.astype(np.int32)
    
    print(f"Tracking {images_array.shape[0]} frames with shape {images_array.shape[1:]}...")
    
    # Load a pretrained model
    model = Trackastra.from_pretrained("general_2d", device=device)

    # Track the cells - pass numpy arrays
    track_graph = model.track(images_array, masks_array, mode="ilp")  # or mode="ilp", or "greedy_nodiv"

    # Write to cell tracking challenge format
    ctc_tracks, masks_tracked = graph_to_ctc(
        track_graph,
        masks_array,  # Use the array version
        #outdir="tracked",
    )
    napari_tracks = graph_to_napari_tracks(track_graph)

    return track_graph, ctc_tracks, masks_tracked, napari_tracks


def load_tracks_data(savedir, indicator, graph=True, ctc=True, masks=True, imgs=False):
    """
    Load tracks data, masks and images related to a timelapse from a specified directory.
    
    Parameters:
    savedir (str): Directory where the file is saved.
    indicator (str): Unique indicator of the file to load - must be contained in track graph, ctc track, imgs, and masks.
    graph (bool): Whether to load the tracks graph. Default is True.
    ctc (bool): Whether to load the cell tracking data. Default is True.
    masks (bool): Whether to load the tracked masks. Default is True.
    imgs (bool): Whether to load the images of the timelapse. Default is False.
    
    Returns:
    tracks_graph (nx.classes.digraph): Loaded tracks graph.
    tracks_ctc (pd.DataFrame): DataFrame containing the cell tracking data.
    masks (np.ndarray): Tracked masks of the tracks.
    imgs (np.ndarray): Images of the timelapse.
    """
    graph_filename = f"tracked_graph_image_{indicator}"
    ctc_filename = f"tracked_masks_image_{indicator}.csv"
    masks_filename = f"tracked_masks_image_{indicator}.npy"
    imgs_filename = f"image_{indicator}.npy"
    loader = ()

    try:
        with open(f"{savedir}/{graph_filename}", 'rb') as f:
            tracks_graph = pickle.load(f)
            loader = loader + (tracks_graph,)
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file {graph_filename} not found in {savedir}.")
    except pickle.UnpicklingError:
        raise ValueError(f"Error unpickling the graph file {graph_filename}.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the graph: {e}")
    try:
        tracks_ctc = pd.read_csv(f"{savedir}/{ctc_filename}")
        loader = loader + (tracks_ctc,)
    except FileNotFoundError:
        raise FileNotFoundError(f"CTC file {ctc_filename} not found in {savedir}.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CTC file {ctc_filename} is empty.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the CTC file: {e}")
    try:
        masks = np.load(f"{savedir}/{masks_filename}")
        loader = loader + (masks,)
    except FileNotFoundError:
        raise FileNotFoundError(f"Masks file {masks_filename} not found in {savedir}.")
    except ValueError:
        raise ValueError(f"Error loading masks file {masks_filename}. Ensure it is a valid .npy file.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the masks file: {e}")
    try:
        imgs = np.load(f"{savedir}/{imgs_filename}")
        loader = loader + (imgs,)
    except FileNotFoundError:
        raise FileNotFoundError(f"Images file {imgs_filename} not found in {savedir}.")
    except ValueError:
        raise ValueError(f"Error loading images file {imgs_filename}. Ensure it is a valid .npy file.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the images file: {e}")
    
    return loader

def load_tracks_data_gui():
    """
    GUI for loading tracks data, masks, and images related to a timelapse.
    
    Returns:
    tracks_graph (nx.classes.digraph): Loaded tracks graph.
    tracks_ctc (pd.DataFrame): DataFrame containing the cell tracking data.
    masks (np.ndarray): Tracked masks of the tracks.
    imgs (np.ndarray): Images of the timelapse.
    indicator (str): Unique indicator of the file to load.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    savedir = filedialog.askdirectory(title="Select Directory with Tracks Data")
    
    if not savedir:
        messagebox.showerror("Error", "No directory selected.")
        return None
    
    indicator = simpledialog.askstring(title="Input", prompt="Enter the unique indicator for the file to load: ")
    root.destroy()  # Close the root window
    try:
        tracks_graph, tracks_ctc, masks, imgs = load_tracks_data(savedir, indicator, imgs=True)
        return tracks_graph, tracks_ctc, masks, imgs, indicator
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None
    
def has_branching(subgraph):
    """
    Check if a subgraph has branching, i.e., if it has more than one outgoing edge from any node.
    
    Parameters:
    subgraph (nx.classes.digraph): The subgraph to analyze.
    
    Returns:
    bool: True if the subgraph has branching, False otherwise.
    """
    for node, degree in subgraph.out_degree():
        if degree > 1:
            return True
    return False
def check_true_parent(parent, lineage, masks, window=3):
    """
    Check if a node is a true parent node in the lineage graph.
    
    Parameters:
    parent (int): The node to check.
    lineage (nx.classes.digraph): The lineage graph to check against.
    masks (np.ndarray): Masks of the tracked objects, used for region properties.
    window (int): The size of the window to check for properties. Default is 3.

    Note:
    The function checks if the parent node has any ancestors in the lineage graph and if it has any descendants.
    A true parent node should have a low eccentricity and high solidity (be round and compact).
    However, if the cell splitting is slow, the dublet state can be mistaken for a parent node.
    In the dublet state, eccentricity is higher, and the parent node would wrongly be considered an artificial parent.
    
    Returns:
    bool: True if the node is a true parent, False otherwise.
    parent (int): The parent node - can be the parent supplied or an ancestor of the parent node.
    props (skimage.measure RegionProperties object): An object with the properties of the parent node.
    """
    # Check if the parent has any ancestors in the lineage graph
    ancestors = list(nx.ancestors(lineage, parent)) + [parent]
    # Get the label and timeframe of the parent node
    timeframe = lineage.nodes[parent]['time']
    window = min(window, timeframe, len(ancestors)-1)  # Ensure the window does not exceed the timeframe or number of ancestors
    # Get the props for the timeframe and the window
    props = [measure.regionprops((m==lineage.nodes[p]['label']).astype(int)) for m,p in zip(masks[timeframe-window:timeframe+1],ancestors[-window-1:]) if m is not None]
    # Get props to evaluate the parent node
    eccentricity = [p[0].eccentricity for p in props if p is not None and len(p) > 0]
    solidity = [p[0].solidity for p in props if p is not None and len(p) > 0]
    criterion = [(lambda x,y: True if x<0.45 and y>0.8 else False)(ecc, sol) for ecc, sol in zip(eccentricity, solidity)]
    if any(criterion):
        # If any of the props meet the criterion, the parent is a true parent
        # Get the index of the first timepoint where the criterion is met
        index = criterion.index(True)
        # Get the parent node at that timepoint
        parent = ancestors[-index] 
        return True, parent, props[-index][0]  # Return the parent node and its properties
    else:
        # If no props meet the criterion, the parent is not a true parent
        return False, parent, props[-1][0]

def sort_parents(subgraph, masks, lineage) -> list:
    """
    Determine the main parent node in a subgraph.
    
    Parameters:
    subgraph (nx.classes.digraph): The subgraph to analyze.
    masks (np.ndarray): Masks of the tracked objects, used for region properties.
    lineage (nx.classes.digraph): The lineage graph to check for true parent nodes.

    Returns:
    list(main_parent (touple), artificial_parents (touples)): List of touples with
                                                                     parent nodes, 
                                                                     their region properties, 
                                                                     and the number of descendants.
                        The main parent is the first node, the following nodes are the artificial parents reachable from the main.
    
    None if no true parent nodes are found.
    """
    parent_nodes = [node for node, degree in subgraph.out_degree() if degree > 1]
    if len(parent_nodes) == 0:
        # No parent nodes found
        return None
    candidates = []  # List to store candidate parent nodes
    artificial_parents_candidates = []  # List to store candidate artificial parent nodes
    # Get regionprops of the parent nodes
    for parent in parent_nodes:
        isParent, parent, props = check_true_parent(parent, lineage, masks)

        if isParent:
            # Likely a true parent node
            candidates.append((parent,props, len(list(nx.descendants(lineage, parent)))))
        else:
            # Likely an artificial parent node
            artificial_parents_candidates.append((parent,props, len(list(nx.descendants(lineage, parent)))))
    
    if len(candidates) == 0:
        # No candidates found
        return None
    elif len(candidates)==1:
        # Only one candidate, return it
        main_parent = candidates[0]    
    elif len(candidates) > 1:
        # Multiple candidates 
        # Sort by the number of descendants
        #candidates.sort(key=lambda x: x[2], reverse=True)
        # Sort by eccentricity low to high
        candidates.sort(key=lambda x: (x[1].eccentricity))
        # Return the first candidate
        main_parent = candidates[0]
 
    # Get the parents reachable from the main parent
    artificial_parents = [n for n in artificial_parents_candidates if n[0] in nx.descendants(subgraph, main_parent[0])]
    return main_parent, artificial_parents  # Only one candidate, return it

def match_source_target(source, target, lineage, masks):
    """
    Link the source and target nodes in a lineage graph.
    
    Parameters:
    source (list): List of source nodes - two source nodes.
    target (list): List of target nodes - two target nodes.
    lineage (nx.classes.digraph): The lineage graph to modify.
    masks (ndarray): Masks array to fix
    
    Returns:
    none: The function modifies the lineage graph in place.
    """
    # Get the coordinates of the source nodes
    source_coords = [lineage.nodes[s]['coords'] for s in source]
    dest_coords = [lineage.nodes[t]['coords'] for t in target]
    distances = np.zeros((len(source_coords), len(dest_coords)))
    # Calculate the distance between each source and target node
    for i, s in enumerate(source_coords):
        for j, d in enumerate(dest_coords):
            # Get distance between source and target nodes
            distance = np.linalg.norm(np.array(s) - np.array(d))
            distances[i, j] = distance
    # Add columns in the distances matrix for the source and target nodes
    costs = distances[0,:] + distances[1, :]
    # Find the minimum cost
    min_cost_index = np.argmin(costs)
    if min_cost_index == 1:
        # The first source node matches the second target node
        target[0], target[1] = target[1], target[0]  # Swap the target nodes
    if (np.any(source==1077) or np.any(target==1077)):
        pass
    # Update the target nodes in the lineage graph
    add_links(source, target, lineage)


def add_links(source, target, lineage):
    """
    Add links between source and target nodes in a lineage graph.
    
    Parameters:
    source (list): List of source nodes - two source nodes.
    target (list): List of target nodes - two target nodes.
    lineage (nx.classes.digraph): The lineage graph to modify.
    
    Returns:
    None
    """
    # Add the links between the source and target nodes
    for s, t in zip(source, target):
        if not lineage.has_edge(s, t):
            lineage.add_edge(s, t)
        # Update descendants' labels to match the source node - this must be done after the lineage is fixed
        # new_label = lineage.nodes[s]['label']
        # descendants = set(nx.descendants(lineage, t)) | {t}
        # for d in descendants:
        #     old_label = lineage.nodes[d]['label']
        #     masks[lineage.nodes[d]['time']][masks[lineage.nodes[d]['time']] == old_label] = new_label
        #     lineage.nodes[d]['label'] = new_label

def try_resegmentation(layer, lineage, masks, imgs):
    """
    Try to resegment the layer in the lineage graph.
    
    Parameters:
    layer (list): List with a single node in the layer that lost daughter.
    lineage (nx.classes.digraph): The lineage graph to modify.
    
    Returns:
    success (bool): Whether two cells are successfully segmented. The function modifies the lineage graph and masks in place.
    """
   # Get the image of the node in the layer
    # label = subgraph.nodes[layer[0]]['label']
    # timeframe = subgraph.nodes[layer[0]]['time']
    # mask = masks[timeframe] != label
    # image = imgs[timeframe]
    # image.where(not mask)==0  # Apply the mask to the image
    # props = measure.regionprops(mask.astype(int))
    return False  # Placeholder for resegmentation logic
def print_lineage(lineage, main_parent=None):
    """
    Print the lineage graph in a readable format.
    
    Parameters:
    lineage (nx.classes.digraph): The lineage graph to print.
    main_parent (int, optional): The main parent node from which to start printing the lineage. If none then root is used
    
    Returns:
    None
    """
    if main_parent is None:
        main_parent = [n for n, d in lineage.in_degree() if d == 0][0]
    layers = list(nx.bfs_layers(lineage, main_parent))
    print('Lineage start:-----------------------------------')
    print("Layer\tTime node 1\tTime node 2\tNodes \t\tLabels")
    for i, layer in enumerate(layers):
        if len(layer)==1:
            print(f"layer {i}: \t{lineage.nodes[layer[0]]['time']} \t \t\t{layer} \t\t{lineage.nodes[layer[0]]['label']}")
        elif len(layer)==2:
            print(f"layer {i}: \t{lineage.nodes[layer[0]]['time']} \t {lineage.nodes[layer[1]]['time']} \t\t{layer}\t{lineage.nodes[layer[0]]['label']},{lineage.nodes[layer[1]]['label']}") 
            
def fix_lineage(lineage, main_parent, masks=None, imgs=None):
    """
    Fix the lineage graph by removing artificial parent nodes and maintaining connections.
    
    Parameters:
    lineage (nx.classes.digraph): The lineage graph to modify.
    main_parent (int): The main parent node from which to start fixing the lineage.
    masks (np.ndarray): Masks of the tracked objects, used for resegmentation if needed.
    imgs (np.ndarray): Images of the timelapse, used for resegmentation if needed.
    
    Returns:
    None: The function modifies the lineage graph in place.
    """
    # Get layers view of the lineage
    layers = list(nx.bfs_layers(lineage, main_parent))
    #print_lineage(lineage, main_parent)

    # Go through the graph layer by layer and find sources, targets and nodes to be removed
    sources = []
    targets = []
    nodes_to_remove = []
    record = False  # Flag to indicate if we are recording nodes to remove
    len_previous_layer = 0
    last_layer = []  # Prev layer in the lineage
    for i, layer in enumerate(layers[1:]):
        len_layer = len(layer)
        if len_layer ==1 and len_previous_layer==2:
            # This is the first timepoint where daughter is lost
            # Try to resegment. If it doesn't result in 2 cells, previous layer is a source
            # Start recording nodes to remove
            if not try_resegmentation(layer, lineage, masks, imgs):
                sources.append(last_layer)
                record = True
        elif len_layer==2 and len_previous_layer==1 and record==True:
            # If two daughters are found again (unlikely to have two real divisions so close)
            # This is the target layer to match sources on
            # Store layer and stop recording nodes to remove
            targets.append(layer)
            record = False
        if record:
            nodes_to_remove.extend(layer)
        len_previous_layer = len_layer
        last_layer = layer  # Update the last layer to the current one
    
    # Check each source layer has a target layer to be linked with
    if len(sources) != len(targets):
        if len(sources) > len(targets):
            # More sources than targets, remove the extra sources
            sources = sources[:len(targets)]
            
        elif len(targets) > len(sources):
            # More targets than sources, remove the extra targets
            # This should not happen, but if it does, we will remove the extra targets
            raise RuntimeError(f"Found more targets than errors.")
            #targets = targets[:len(sources)]
    
    # Make the links from sources to targets
    for source, target in zip(sources, targets):
        # Match the source and target nodes
        match_source_target(source, target, lineage, masks)
        
    # Remove the artificial parents from the lineage
    lineage.remove_nodes_from(nodes_to_remove)
    # Check lineage
    #print_lineage(lineage, main_parent)
def save_lineage_snapshot(lineage, masks, imgs, identifier="lineage_snapshot"):
    """
    Save a snapshot of the lineage graph, masks, and images.
    
    Parameters:
    lineage (nx.classes.digraph): The lineage graph to save.
    masks (np.ndarray): Masks of the tracked objects.
    imgs (np.ndarray): Images of the timelapse.
    
    Returns:
    None: The function saves the lineage graph, masks, and images to files.
    """
    # Get lineage root node
    root = [n for n, d in lineage.in_degree() if d == 0]
    # Get the layers of the lineage graph
    layers = list(nx.bfs_layers(lineage,sources=root[0]))
    # Go through the lineage layer by layer and collect the bboxes of the nodes. 
    # Use the bboxes to create a snapshot of the lineage.
    snapshot = []
    mask_snapshot = []
    for layer in layers: 
        # Get the properties of the nodes in the layer
        # calculate the bounding box that contains all the nodes in the layer
        # and save the image and mask of the layer  
        layer_bboxes = [] 
        masks_layer = np.zeros_like(masks[0])  # Create a mask for the layer
        timeframe = lineage.nodes[layer[0]]['time']
        for node in layer:
            props = get_node_props(node, lineage, masks, imgs)
            if props is not None:
                lineage.nodes[node]['props'] = props
                layer_bboxes.append(props.bbox)
                masks_layer=masks_layer + lineage.nodes[node]['label']*props._label_image
        if len(layer_bboxes) == 0:
            continue
        elif len(layer_bboxes) == 1:
            # Only one node in the layer, use its bbox
            bbox = layer_bboxes[0]
        elif len(layer_bboxes) > 1:
            # More than one node in the layer, calculate the bounding box that contains all the nodes in the layer
            min_row = min(bbox[0] for bbox in layer_bboxes)
            min_col = min(bbox[1] for bbox in layer_bboxes)
            max_row = max(bbox[2] for bbox in layer_bboxes)
            max_col = max(bbox[3] for bbox in layer_bboxes)
            bbox = (min_row, min_col, max_row, max_col)
        # Get the image and mask of the layer
        snapshot.append(imgs[timeframe, bbox[0]:bbox[2], bbox[1]:bbox[3]])
        # Get the masks of the nodes in the layer
        mask_snapshot.append(masks_layer[bbox[0]:bbox[2], bbox[1]:bbox[3]])
    mask_snapshot = homogenize_list(mask_snapshot,0)
    snapshot = homogenize_list(snapshot,90)
    with open(f"{identifier}_lineage_snapshot.pkl", "wb") as f:
        pickle.dump(lineage, f)
    np.save(f"{identifier}_masks_snapshot.npy", np.asanyarray(mask_snapshot))
    np.save(f"{identifier}_imgs_snapshot.npy", np.asarray(snapshot))
    print("Lineage snapshot saved.")

def save_lineage_masks(lineage, masks, identifier="lineage_snapshot"):
    """
    Save a snapshot of the lineage graph and masks to fit on original image
    
    Parameters:
    lineage (nx.classes.digraph): The lineage graph to save.
    masks (np.ndarray): Masks of the tracked objects.
    imgs (np.ndarray): Images of the timelapse.
    identifier (string): Text to add to the names of the files
    
    Returns:
    None: The function saves the lineage graph, masks, to files.
    """
    # Get lineage root node
    root = [n for n, d in lineage.in_degree() if d == 0]
    # Get the layers of the lineage graph
    layers = list(nx.bfs_layers(lineage,sources=root[0]))
    # Go through the lineage layer by layer and collect the masks with label == node label
    masks_snapshot = np.zeros_like(masks)
    for layer in layers: 
        for node in layer:
            timeframe = lineage.nodes[node]['time']
            if timeframe==161:
                pass
            label=lineage.nodes[node]['label']
            masks_snapshot[timeframe]=masks_snapshot[timeframe] + label*(masks[timeframe]==label)
        
    with open(f"{identifier}_graph.pkl", "wb") as f:
        pickle.dump(lineage, f)
    print(f"Saved {identifier}")
    print_lineage(lineage,root[0])
    np.save(f"{identifier}_masks.npy", np.asanyarray(masks_snapshot))
    
def homogenize_list(snapshot,background):
    """
    Make the array elements of a list homogenous in size

    Parameters:
    snapshot (list): list of ndarrays 
    background (int): value to use for fill-in

    Returns:
    snapshot (ndarray)

    """
    rows_size = [s.shape[0] for s in snapshot]
    columns_size = [s.shape[1] for s in snapshot]

    r_max = max(rows_size)
    c_max = max(columns_size)

    shift_r = [int((r_max-rs)/2) for rs in rows_size]
    shift_c = [int((c_max-cs)/2) for cs in columns_size]
    snaph=background*np.ones((len(snapshot),r_max,c_max))
    for i, snap in enumerate(snapshot):
        rs = snap.shape[0]
        cs = snap.shape[1]
        shift_r=int((r_max-rs)/2)
        shift_c=int((c_max-cs)/2)
        snaph[i,shift_r:shift_r+rs,shift_c:shift_c+cs] = snap
    return snaph


def get_node_props(node, lineage, masks, imgs):
    """
    Get the properties of a node in the lineage graph.
    
    Parameters:
    node (int): The node to get properties for.
    lineage (nx.classes.digraph): The lineage graph.
    masks (np.ndarray): Masks of the tracked objects.
    imgs (np.ndarray): Images of the timelapse.
    
    Returns:
    props (skimage.mneasure RegionProperties object): An object with the properties of the node.
    """
    timeframe = lineage.nodes[node]['time']
    label = lineage.nodes[node]['label']
    mask = masks[timeframe] == label
    img = imgs[timeframe]
    props = measure.regionprops(mask.astype(int),img)
    return props[0] if props else None
def fix_masks(lineage, masks):
    """
    After lineage links were fixed, go through the graph and propagate labels.

    Parameters:
    lineage (nx.classes.digraph): The fixed lineage
    masks(np.ndarray): The masks to be fixed in place

    Returns:
    None
    """
    # Get lineage root node:
    root = [n for n, d in lineage.in_degree() if d == 0]
    layers = nx.bfs_layers(lineage, root[0])
    for layer in layers:
        for node in layer:
            # If only one outgoing link from node then propagate its label
            if lineage.out_degree(node) == 1:
                #Assign its label to the following node
                child = next(lineage.successors(node))
                label = lineage.nodes[node]['label']
                timeframe = lineage.nodes[child]['time']
                masks[timeframe][masks[timeframe] == lineage.nodes[child]['label']] = label
                lineage.nodes[child]['label'] = label

def rank_lineages(lineages, imgs):
    """
    Rank the lineages based on some criteria.
    Parameters:
    lineages (list): List of lineage graphs to rank.
    imgs (np.ndarray): Images/masks of the timelapse.
    Returns:
    ranked_indices (ndarray): Indices of the lineages sorted by length and distance to the center of the image.
    """
    # Sort lineages based on length and root centroid distance to the middle of the image
    avg_distances = []
    lengths =[]
    center_coords = np.array(imgs.shape[1:]) / 2  # Assuming imgs is a 3D array (time, height, width)
    for lineage in lineages:
        # Calculate the average centroid distance to the middle of the image
        # centroids = [np.array(lineage.nodes[n]['coords']) for n in lineage.nodes]
        # distances = [np.linalg.norm(center_coords - c) for c in centroids]
        # Calculate the distance of the parent node to the center of the image
        parent_node = [n for n, d in lineage.out_degree() if d == 2][0]
        parent_coords = np.array(lineage.nodes[parent_node]['coords'])
        distance = np.linalg.norm(center_coords - parent_coords)
        avg_distances.append(distance)
        lengths.append(len(lineage.nodes))
    # Get ranked indices based on average distances
    dist_ranked_indices = np.argsort(avg_distances)
    length_ranked_indices = np.flipud(np.argsort(lengths))  # Sort lengths in descending order
    # Create a rank array for distances and lengths
    dist_rank =np.zeros_like(dist_ranked_indices, dtype=int)  # Initialize rank array
    length_rank = np.zeros_like(dist_ranked_indices, dtype=int)  # Initialize length rank array

    for i, idx in enumerate(zip(dist_ranked_indices, length_ranked_indices)):
        dist_rank[idx[0]] = i  # Assign ranks based on sorted order
        length_rank[idx[1]] = i  # Assign ranks based on sorted order
    # Combine the ranks to get a final ranking
    combined_rank = 3*dist_rank + length_rank
    ranked_indices = np.argsort(combined_rank)

    return ranked_indices

if __name__ == "__main__":
    tracks_graph, tracks_ctc, masks, imgs, indicator = load_tracks_data_gui()
    lineages = []  # List to store the lineages found in the tracks graph
    if tracks_graph is not None:
        print("Tracks graph loaded successfully.")
        print("Tracks CTC data loaded successfully.")
        print("Masks loaded successfully.")
        print("Images loaded successfully.")
        # Get total number of subgraphs in the tracks graph
        n_subgraphs = nx.number_weakly_connected_components(tracks_graph)

        # Get the subgrgraphs of the tracks graph
        for i, subgraph_iter in enumerate(nx.weakly_connected_components(tracks_graph)):
            #print(f"Processing subgraph {i} of {n_subgraphs}...")
            # Create a read-only subgraph from the current component
            subgraph = tracks_graph.subgraph(subgraph_iter)
            #print(f"Subgraph with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges.")
            # Check if there is any branching in the subgraph
            if has_branching(subgraph):
                #This subgraph has branching - parent-child relationships
                #Check if there are any true parent nodes
                sorted_parents= sort_parents(subgraph, masks, tracks_graph)
                if sorted_parents is None:
                    #print("No true parent nodes found in this subgraph.")
                    continue
                else:
                    main_parent, artificial_parents = sorted_parents
                    lineage_nodes = set(nx.descendants(tracks_graph, main_parent[0])) | {main_parent[0]}
                    lineage = tracks_graph.subgraph(lineage_nodes).copy()
                    lineages.append(lineage)
                    print(f"Found a division - lineage {len(lineages)-1}")
                    #print_lineage(lineage, main_parent[0])
                    if len(artificial_parents) > 0:
                        print(f"Found artificial parents - lineage {len(lineages)-1}")
                    #Fix lineage by removing the artificial parents from the lineage maintaining connections
                    fix_lineage(lineage, main_parent[0])
                    fix_masks(lineage, masks)

            else:
                #print("This subgraph does not have branching.")
                continue
        print(f"Total number of lineages found: {len(lineages)}")
        # Rank lineages
        ranks = rank_lineages(lineages, imgs)
        # Save the lineage with the highest rank
        highest_ranked_lineage = lineages[ranks[0]]
        print(f"Highest ranked lineage: {ranks[0]}")
        # Save the highest ranked lineage
        save_lineage_masks(highest_ranked_lineage, masks, identifier=f"highest_ranked_lineage_{ranks[0]}_{indicator}")
        #Save the lineages to a file
        for i, lineage in enumerate(lineages):
            identifier = f"img_{indicator}_lineage-{i}"
            save_lineage_masks(lineage, masks, identifier)
    
        
    else:
        print("Failed to load tracks data.")