import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
from utils import convert_to_minimal_format
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename, askopenfilename
from tkinter import messagebox
from tkinter.simpledialog import askstring
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import matplotlib
from experiment_statistics import ExperimentStatistics
from experiment_plots import ExperimentPlots
from sklearn.decomposition import PCA
matplotlib.use('TkAgg')
PROPERTIES_ALL = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std','bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')
PROPERTIES_MINIMAL = ('label', 'area', 'perimeter', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std')
PROPERTIES_NO_INTENSITY = ('label', 'area', 'perimeter', 'centroid', 'bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')
PROPERTIES_CELL_SHAPE = ('label', 'area', 'perimeter', 'centroid', 'bbox', 'eccentricity', 'solidity', 'orientation', 'major_axis_length', 'minor_axis_length')
PROPERTIES_INTENSITY = ('label', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std')
FIG_SIZE = (3,2) # Width, Height in inches
# Class to hold experiment sample data and metadata
# contains name, description, dask arrays for the images and masks, and a pandas dataframe holding region properties
class ExperimentSample:
    def __init__(self, name, group, description = None, imagePath = None, masksPath = None, regionprops_df_path=None, bitDepth=16, normalize = False):
        self.name = name
        self.group = group
        self.description = description
        self.imagePath = imagePath
        self.masksPath = masksPath
        if regionprops_df_path:
            self.regionprops_df = pd.read_csv(regionprops_df_path)
        else:
            self.regionprops_df = None
        self.bitDepth = bitDepth
        if self.regionprops_df is None and self.imagePath is not None and self.masksPath is not None:
            self.compute_regionprops()
        if self.regionprops_df is not None and normalize:
            self.normalize_intensity()
    def load_regionprops(self, path=None):
        if path is None:
            path = askopenfilename(title="Path for regionprops csv:",filetypes=[("CSV files", "*.csv")])
        self.regionprops_df = pd.read_csv(path)
    def load_image(self, path=None):
        if path is None:
            path = askopenfilename(title="Path for image:", filetypes=[("Image files", "*.png;*.jpg;*.tif;*.npy; *.lif; *.czi; *.nd2")])
        self.imagePath = path
    def load_masks(self, path=None):
        if path is None:
            path = askopenfilename(title="Path for masks:", filetypes=[("Image files", "*.png;*.jpg;*.tif;*.npy; *.lif; *.czi; *.nd2")])
        self.masksPath = path
    def set_group(self, group=None):
        if group is None:
            group = askstring("Input", f"Set group for sample {self.name}:", parent=Tk())
        self.group = group
    def compute_regionprops(self, properties=PROPERTIES_ALL):
        from aicsimageio import AICSImage
        from skimage.measure import regionprops_table
        # Check if image and mask paths are set
        if not self.imagePath or not self.masksPath:
            raise ValueError("Image and masks paths must be set to compute region props")
        # Compute region properties for the sample
        if self.imagePath.endswith(".npy"):
            image_data = np.load(self.imagePath)
        else:
            img = AICSImage(self.imagePath)
            image_data = img.get_image_dask_data("YXC")
        if self.masksPath.endswith(".npy"):
            mask_data = np.load(self.masksPath)
        else:
            mask = AICSImage(self.masksPath)
            mask_data = mask.get_image_dask_data("YX")
        # Check that dimensions match
        if image_data.shape[:-1] != mask_data.shape:
            print("Image and mask dimensions do not match. Will crop image to match mask.")
            image_data = image_data[..., :mask_data.shape[0], :mask_data.shape[1]]

        # Compute region properties
        props = regionprops_table(mask_data, intensity_image=image_data, properties=properties)
        self.regionprops_df = pd.DataFrame(props)
    
    def normalize_intensity(self):
        # Normalize intensity values to the range [0, 1]
        if 'intensity_mean' in self.regionprops_df.columns:
            self.regionprops_df['intensity_mean'] /= (2 ** self.bitDepth - 1)
        if 'intensity_min' in self.regionprops_df.columns:
            self.regionprops_df['intensity_min'] /= (2 ** self.bitDepth - 1)
        if 'intensity_max' in self.regionprops_df.columns:
            self.regionprops_df['intensity_max'] /= (2 ** self.bitDepth - 1)
        if 'intensity_std' in self.regionprops_df.columns:
            self.regionprops_df['intensity_std'] /= (2 ** self.bitDepth - 1)

    def save_regionprops(self, path=None):
        if path is None:
            path = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        self.regionprops_df.to_csv(path, index=False)


class RegionpropsFilter:
    def __init__(self, column, type, thresholds_low=None, thresholds_high=None, category_remove=None, category_keep=None):
        """
        Initialize a RegionpropsFilter. 

        Args:
            column (str): The column to filter on.
            type (str): The type of filter. Must be either "numeric" or "categorical".
            thresholds_low (list, optional): The lower threshold for filtering. Anything < threshold_low is filtered out.
            thresholds_high (list, optional): The upper threshold for filtering. Anything > threshold_high is filtered out.
            category_remove (str, optional): The category to filter by. Anything in the category is filtered out.
            category_keep (str, optional): The category to keep. Anything not in the category is filtered out.
        """
        self.column = column
        self.type = type
        self.thresholds_low = thresholds_low
        self.thresholds_high = thresholds_high
        self.category_remove = category_remove
        self.category_keep = category_keep

# Class to hold multiple experiment samples with functions to combine regionprops dataframes, compute summary statistics
# and plot population plots grouped by group with seaborn
class Experiment:
    def __init__(self):
        self.samples = []
        self.regionprops = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.scatter_plot = None
        self.cat_plot = None
        self.joint_plot = None # Not implemented
        self.pair_plot = None
        self.figSize = FIG_SIZE
        self.statistics = ExperimentStatistics(self)  # Use composition
        self.plots = ExperimentPlots(self, self.regionprops)  # Use composition

    def add_sample(self, sample):
        self.samples.append(sample)
        sampleDf = sample.regionprops_df
        # Add column to identify sample
        sampleDf['sample'] = sample.name
        sampleDf['group'] = sample.group
        # Add to Experiment repeats dataframe
        self.regionprops = pd.concat([self.regionprops, sampleDf], ignore_index=True)

    def remove_sample(self, sample_name):
        self.samples = [s for s in self.samples if s.name != sample_name]
        self.regionprops = self.regionprops[self.regionprops['sample'] != sample_name]
        if self.summary is not None:
            # Update summary statistics
            self.summary = self.summarize()
    
    def add_group(self, group_name=None, samples = None):
        if group_name is None:
            group_name = askstring(title="Add group", prompt="Set group name")
        if samples is None:
            paths = askopenfilenames(title="Select sample regionprops csv files for group " + group_name, filetypes=[("CSV files", "*.csv")])
            samples = [ExperimentSample(regionprops_df_path=path, group=group_name, name=os.path.basename(path).split('.')[0]) for path in paths]
        for sample in samples:
            self.add_sample(sample)

    def remove_group(self, group_name):
        self.regionprops = self.regionprops[self.regionprops['group'] != group_name]
        if self.summary is not None:
            self.summary = self.summarize()

    def summarize(self):
        # Only use numeric columns for aggregation
        numeric_cols = self.regionprops.select_dtypes(include=[np.number]).columns
        self.summary = self.regionprops.groupby('group')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def normalize_intensity(self, camera_bit_depth):
        # Normalize intensity values to the range [0, 1]
        # Check the "intensity_" columns exist
        if 'intensity_mean' in self.regionprops.columns:
            self.regionprops['intensity_mean'] /= (2 ** camera_bit_depth - 1)
        if 'intensity_min' in self.regionprops.columns:
            self.regionprops['intensity_min'] /= (2 ** camera_bit_depth - 1)
        if 'intensity_max' in self.regionprops.columns:
            self.regionprops['intensity_max'] /= (2 ** camera_bit_depth - 1)
        if 'intensity_std' in self.regionprops.columns:
            self.regionprops['intensity_std'] /= (2 ** camera_bit_depth - 1)

    def filter_data(self, filters):
        """
        Filter the regionprops dataframe based on thresholds on any column.

        Args:
            filters (list of RegionpropsFilter): List of filter objects to apply.
        """
        for filter in filters:
            if filter.column in self.regionprops.columns:
                if filter.type == 'categorical':
                    if filter.category_remove is not None:
                        self.regionprops_filtered = self.regionprops[self.regionprops[filter.column] != filter.category_remove]
                    if filter.category_keep is not None:
                        self.regionprops_filtered = self.regionprops[self.regionprops[filter.column] == filter.category_keep]
                elif filter.type == 'numeric':
                    if filter.thresholds_low is not None:
                        self.regionprops_filtered = self.regionprops[self.regionprops[filter.column] >= filter.thresholds_low]
                    if filter.thresholds_high is not None:
                        self.regionprops_filtered = self.regionprops[self.regionprops[filter.column] <= filter.thresholds_high]

    def normalize_regprops_across_groups(self, columns):
        # Check columns are numeric
        numeric_columns = self.regionprops.select_dtypes(include=[np.number]).columns
        columns = [col for col in columns if col in numeric_columns]
        # Normalize regionprops across groups
        self.regionprops[columns] = self.regionprops[columns].groupby(self.regionprops['group']).transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)

    def get_common_attributes(self):
        # Get the common attributes across all samples
        common_attrs = set.intersection(*(set(s.regionprops_df.columns) for s in self.samples))
        return common_attrs

    def plot_jointplot(self, x, y, hue='group', kind='joint', show=False, title=None, xlabel=None, ylabel=None, **kwargs):
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
        if kind == 'scatter':
            sns.scatterplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif kind == 'box':
            sns.boxplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif kind == 'joint':
            sns.jointplot(data=self.regionprops, x=x, y=y, hue=hue, kind='kde')
        #ax.set_title(f'Population Plot: {y} vs {x}')
        ax.legend(loc='best', fontsize='small')
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        #plt.xticks(rotation=45, fontsize=8)
        #plt.yticks(fontsize=8)
        self.scatter_plot = fig
        if show:
            plt.show()

    def plot_categorical_comparisons(self, plot_kind, metric, show=False, title=None, xlabel=None, ylabel=None, **kvargs):
        from statannotations.Annotator import Annotator

        sns.set_theme(style="darkgrid", palette="colorblind")
        fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
        fig.tight_layout()
        # Plot types that only accept x or y, not both
        one_axis_plots = {"count", "bar", "point"}
        if plot_kind in one_axis_plots:
            sns_plot = getattr(sns, plot_kind + "plot", sns.boxplot)
            sns_plot(data=self.regionprops, x="group", hue='group', ax=ax)
            ax.set_ylabel("Count" if plot_kind == "count" else metric)
        else:
            sns_plot = getattr(sns, plot_kind + "plot", sns.boxplot)
            sns_plot(data=self.regionprops, x="group", y=metric, hue='group', ax=ax)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Prepare pairs for annotation (all pairwise group comparisons)
        groups = self.regionprops['group'].unique()
        pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]

        # Prepare p-values for annotation
        self.statistics.compare_groups(metric=metric)
        
        if hasattr(self, "tukey_results") and not self.tukey_results.empty:
            pvalues = []
            for g1, g2 in pairs:
                match = self.statistics.tukey_results[
                    ((self.statistics.tukey_results['group1'] == g1) & (self.statistics.tukey_results['group2'] == g2)) |
                    ((self.statistics.tukey_results['group1'] == g2) & (self.statistics.tukey_results['group2'] == g1))
                ]
                if not match.empty:
                    pvalues.append(float(match['p-adj'].values[0]))
                else:
                    pvalues.append(1.0)
            annotator = Annotator(ax, pairs, data=self.regionprops, x="group", y=metric)
            annotator.set_pvalues_and_annotate(pvalues, test_short_name="Tukey", text_format="star", loc="inside", verbose=0)

        ax.set_title(f'{metric} comparison')
        plt.xticks(rotation=90)
        if show:
            plt.show()
        self.cat_plot = fig

    
    def get_principal_components(self, n_components=3, columns=None):
        """ Extracts principal components from regionprops dataframe and returns them.
            Args:
                n_components (int): Number of principal components to return.
                columns (list): List of columns to use for PCA (must be numeric).
            Returns:
                DataFrame with principal components.
        """
        if columns is None:
            columns = self.regionprops.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Remove "group", "label", "index", "sample", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1"
            columns = [col for col in columns if col not in ["group", "label", "index", "sample", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1"]]

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.regionprops[columns])
        return pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])

    def plot_pair_plot(self, dataframe, columns=None, show=False, title=None, xlabel=None, ylabel=None, **kwargs):
        """Plots a pair plot from a regionprops dataframe using group/sample as selectors.

        Args:
            dataframe: DataFrame with regionprops data.
            columns (list): List of columns to include in the plot.
            show (bool): Whether to show the plot.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            **kwargs: Additional keyword arguments for the plot.
        """
        if columns is None:
            # Select numerical columns
            columns = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Remove "group", "label", "index", "sample", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1"
            columns = [col for col in columns if col not in ["group", "label", "index", "sample", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1"]]

        # Create the pair plot
        g = sns.pairplot(data=dataframe, vars=columns, **kwargs)

        # Set titles and labels
        if title is not None:
            g.figure.suptitle(title, fontsize=16)
        
        g.set_axis_labels(xlabel if xlabel else columns[0], ylabel if ylabel else columns[1], fontsize=12)

        if show:
            plt.show()

        self.pair_plot = g.figure

    def save_plots(self):
        """Save all generated plots to files."""
        if self.scatter_plot:
            filename = asksaveasfilename(title="Save Scatter Plot", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                self.scatter_plot.savefig(filename)
        if self.cat_plot:
            filename = asksaveasfilename(title="Save Categorical Plot", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                self.cat_plot.savefig(filename)
                
    def save_regionprops(self, path=None):
        """Save regionprops df to csv"""
        if path is not None:
            self.regionprops.to_csv(path, index=False)
        else:
            path = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            self.regionprops.to_csv(path, index=False)

if __name__ == "__main__":
    def test_add_groups():
        """Test adding groups to the experiment."""
        root = Tk()
        root.withdraw()  # Hide the main window
        
        experiment = Experiment()
        experiment.add_group()
        experiment.add_group()
        
        return experiment

    def test_regionprops():
        im_paths = askopenfilenames(title="Select image files", filetypes=[("Image files", "*.tif *.tiff *.npy"), ("All files", "*.*")])
        mask_paths = askopenfilenames(title="Select mask files", filetypes=[("Image files", "*.tif *.tiff *.npy"), ("All files", "*.*")])
        experiment = Experiment()
        for inx, (im_path, mask_path) in enumerate(zip(im_paths, mask_paths)):
            print(f"Image: {im_path}, Mask: {mask_path}")
            # Create a Sample instance
            sample = ExperimentSample(os.path.basename(im_path).split('.')[0], f"Group{inx+1}", imagePath=im_path, masksPath=mask_path)
            sample.compute_regionprops()
            path = f"{os.path.dirname(im_path)}_regionprops_{sample.name}.csv"
            sample.save_regionprops(path)
            experiment.add_sample(sample)

        path = f"{os.path.dirname(im_paths[0])}_regionprops_full.csv"
        experiment.save_regionprops(path)
        summary = experiment.summarize()
        path = f"{os.path.dirname(im_paths[0])}_regionprops_summary.csv"
        summary.to_csv(path, index=False)
    
    def test_statistics(experiment):
        from experiment_statistics import ExperimentStatistics
        # Load samples with regionprops paths
        experiment.statistics.remove_outliers()
        experiment.statistics.compare_groups(metric='area')
        if hasattr(experiment.statistics, "normality_results"):
            print(f"Normal distributions: {experiment.statistics.normality_results}")
        if hasattr(experiment.statistics,"homoscedasticity_result"):
            print(f"Equal variance: {experiment.statistics.homoscedasticity_result}")
        if hasattr(experiment.statistics, "tukey_results"):
            print(f"Tukey: {experiment.statistics.tukey_results}")
        elif hasattr(experiment.statistics, "anova_results"):
            print(experiment.statistics.anova_results)
        elif hasattr(experiment.statistics, "kruskal_results"):
            print(experiment.statistics.kruskal_results)

    def test_plots(experiment):
        experiment.plot_population(x='area', y='perimeter', hue='group', kind='scatter', show=True)
        experiment.plot_population(x='area', y='perimeter', hue='group', kind='joint', show=True)

    experiment = test_add_groups()
    #test_statistics(experiment)
    test_plots(experiment)