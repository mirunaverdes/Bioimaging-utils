import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
from utils import convert_to_minimal_format
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import matplotlib
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
        self.bitDepth = bitDepth

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

# Class to hold multiple experiment samples with functions to combine regionprops dataframes, compute summary statistics
# and plot population plots grouped by group with seaborn
class Experiment:
    def __init__(self):
        self.samples = []
        self.regionprops = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.scatter_plot = None
        self.cat_plot = None
        self.normality_results = None
        self.homoscedasticity_result = None
        self.figSize = FIG_SIZE

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
    
    def remove_outliers(self):
        # Remove outliers from the regionprops dataframe
        numeric_cols = self.regionprops.select_dtypes(include=[np.number]).columns
        # Exclude 'labels'/'label'/'group'/'sample' column
        numeric_cols = numeric_cols[numeric_cols != 'labels']
        numeric_cols = numeric_cols[numeric_cols != 'label']
        numeric_cols = numeric_cols[numeric_cols != 'group']
        numeric_cols = numeric_cols[numeric_cols != 'sample']
        self.regionprops = self.regionprops[~self.regionprops[numeric_cols].isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        # Find statistical outliers and remove them
        for col in numeric_cols:
            if col in self.regionprops.columns:
                q1 = self.regionprops[col].quantile(0.25)
                q3 = self.regionprops[col].quantile(0.75)
                iqr = q3 - q1
                self.regionprops = self.regionprops[~self.regionprops[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
        # Reset index after removing outliers
        self.regionprops.reset_index(drop=True, inplace=True)

    def normalize_regprops_across_groups(self):
        # Normalize regionprops across groups
        self.regionprops = self.regionprops.groupby('group').transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)

    def get_common_attributes(self):
        # Get the common attributes across all samples
        common_attrs = set.intersection(*(set(s.regionprops_df.columns) for s in self.samples))
        return common_attrs

    def plot_population(self, x, y, hue=None, kind='scatter', show=False, title=None, xlabel=None, ylabel=None, **kwargs):
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
        if kind == 'scatter':
            sns.scatterplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif kind == 'box':
            sns.boxplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
        ax.set_title(f'Population Plot: {y} vs {x}')
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
        self.compare_groups(metric=metric)
        
        if hasattr(self, "tukey_results") and not self.tukey_results.empty:
            pvalues = []
            for g1, g2 in pairs:
                match = self.tukey_results[
                    ((self.tukey_results['group1'] == g1) & (self.tukey_results['group2'] == g2)) |
                    ((self.tukey_results['group1'] == g2) & (self.tukey_results['group2'] == g1))
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
    def compare_groups(self, metric='area'):
        """Make statistical comparisons between groups. Store matrix of p-values of the statistical test"""
        import scipy.stats as stats
        # Test for normality (per group)
        self.normality_results = self.regionprops.groupby('group')[metric].apply(lambda x: self.normality_test(x))
        # Test for homoscedasticity (Levene's test across all groups)
        group_values = [group[metric].dropna().values for name, group in self.regionprops.groupby('group')]
        if len(group_values) < 2:
            self.anova_results = None
            self.kruskal_results = None
            self.tukey_results = pd.DataFrame()
            return
        self.homoscedasticity_result = stats.levene(*group_values)
        # If all groups are normal, and homoscedasticity holds use ANOVA
        if all(result.pvalue > 0.05 for result in self.normality_results) and self.homoscedasticity_result.pvalue > 0.05:
            self.anova_results = stats.f_oneway(*group_values)
            self.posthoc_tukey(metric=metric)
        else:
            # If not all groups are normal or homoscedasticity fails, use Kruskal-Wallis
            self.kruskal_results = stats.kruskal(*group_values)
            self.tukey_results = pd.DataFrame()  # Clear if not applicable

    def posthoc_tukey(self, metric='area'):
        """Run Tukey HSD post hoc test and store results as a DataFrame."""
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        if self.regionprops['group'].nunique() > 1:
            tukey = pairwise_tukeyhsd(endog=self.regionprops[metric], groups=self.regionprops['group'])
            self.tukey_results = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        else:
            self.tukey_results = pd.DataFrame()

    def plot_scatter_data(dataframe, **kwargs):
        """
        Plots scatter data from a regionprops dataframe using group/sample as selectors.

        Args:
            dataframe: DataFrame with regionprops data.
            **kwargs:
                group (str): group name to filter by (must be in dataframe['group'])
                sample (str): sample name to filter by (optional, must be in dataframe['sample'])
                yaxis (str): column name for y-axis
                xaxis (str): column name for x-axis
                yaxis_inx (int): index of y-axis column
                xaxis_inx (int): index of x-axis column
        """
        group = kwargs.get('group')
        if group and group not in dataframe['group'].unique():
            raise ValueError(f"`group` must be one of: {dataframe['group'].unique()}")

        sample = kwargs.get('sample')
        if sample and sample not in dataframe['sample'].unique():
            raise ValueError(f"`sample` must be one of: {dataframe['sample'].unique()}")

        yaxis = kwargs.get('yaxis')
        xaxis = kwargs.get('xaxis')
        yaxis_inx = kwargs.get('yaxis_inx')
        xaxis_inx = kwargs.get('xaxis_inx')

        if yaxis_inx is not None and not yaxis:
            yaxis = dataframe.columns[yaxis_inx]
        if xaxis_inx is not None and not xaxis:
            xaxis = dataframe.columns[xaxis_inx]

        if not yaxis or not xaxis:
            raise ValueError("Both `yaxis` and `xaxis` must be specified (by name or index).")

        # Filter by group and sample if provided
        df = dataframe.copy()
        if group:
            df = df[df['group'] == group]
        if sample:
            df = df[df['sample'] == sample]

        print(f"Group: {group if group else 'All'} | Sample: {sample if sample else 'All'} | Plotting {xaxis} vs {yaxis}")

        sns.set_theme(style="darkgrid", palette="colorblind")
        ax = sns.scatterplot(data=df, x=xaxis, y=yaxis, size='area' if 'area' in df.columns else None, hue='group' if 'group' in df.columns else None)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()

    def normality_test(self, arr):
        import scipy.stats as stats
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 5000:
            return stats.normaltest(arr)
        else:
            return stats.shapiro(arr)

if __name__ == "__main__":
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
    
    def test_statistics():
        # Load samples with regionprops paths
        paths = askopenfilenames(title="Select regionprops CSV files", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        experiment = Experiment()
        for inx, path in enumerate(paths):
            sample = ExperimentSample(os.path.basename(path).split('_regionprops_')[1].split('.csv')[0], f"Group{inx+1}", regionprops_df_path=path)
            experiment.add_sample(sample)
        experiment.remove_outliers()