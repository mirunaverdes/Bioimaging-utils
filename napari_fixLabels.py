from magicgui import magicgui
from napari.layers import Labels
from qtpy.QtWidgets import QFileDialog
import numpy as np

def save_labels_as_npy(labels_layer: Labels = None):
    """Save a napari Labels layer as a .npy file using a file dialog."""
    if labels_layer is None:
        print("No labels layer selected.")
        return
    # Open a file dialog to select save location
    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setNameFilter("NumPy files (*.npy)")
    dialog.setDefaultSuffix("npy")
    if dialog.exec_():
        filename = dialog.selectedFiles()[0]
        np.save(filename, labels_layer.data)
        print(f"Labels layer saved to {filename}")

@magicgui(call_button="Save Labels as .npy", labels_layer={"label": "Labels Layer"})
def save_labels_widget(labels_layer: Labels):
    save_labels_as_npy(labels_layer)

# This function can be used in a napari plugin or script to save labels layers
def register_widget(viewer):
    """Register the save labels widget in the napari viewer."""
    viewer.window.add_dock_widget(save_labels_widget, area='right', name='Save Labels as .npy')
    print("Save Labels widget registered in napari viewer.")

if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    register_widget(viewer)
    napari.run()