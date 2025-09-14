import numpy as np
import pandas as pd
import seaborn as sns
from tkinter.filedialog import asksaveasfilename
from matplotlib import pyplot as plt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from experiment import Experiment

class ExperimentPlots:
    def __init__(self, experiment:"Experiment" = None, regionprops: pd.DataFrame = None, fig_size=(10, 6)):
        self.experiment: "Experiment" = experiment
        self.regionprops = regionprops
        self.fig_size = fig_size
        self.joint_plot: sns.JointGrid = None
        self.pair_plot: sns.PairGrid = None
        # self.relation_plot: sns.FacetGrid = None
        # self.distribution_plot: sns.FacetGrid = None
        self.categorical_plot: sns.FacetGrid = None

    def set_joint_plot(self, x, y, hue='group', kind='kde', show=False, **kwargs):
        df = self.regionprops
        self.joint_plot = sns.jointplot(data=df, x=x, y=y, hue=hue, kind=kind, **kwargs)
        self.style_joint_plot()
        if show:
            plt.show()

    def get_joint_plot(self):
        return self.joint_plot

    def style_joint_plot(self, title=None, xlabel=None, ylabel=None):
        if self.joint_plot is not None:
            # Use direct attribute access instead of .set()
            self.joint_plot.figure.set_size_inches(self.fig_size)
            
            if title:
                self.joint_plot.figure.suptitle(title, fontsize=14)
            if xlabel:
                self.joint_plot.ax_joint.set_xlabel(xlabel, fontsize=12)
            if ylabel:
                self.joint_plot.ax_joint.set_ylabel(ylabel, fontsize=12)
                
            # Style the tick labels
            plt.setp(self.joint_plot.ax_joint.get_xticklabels(), fontsize=10)
            plt.setp(self.joint_plot.ax_joint.get_yticklabels(), fontsize=10)
            
            # Handle legend if it exists
            if self.joint_plot.ax_joint.get_legend():
                self.joint_plot.ax_joint.legend(fontsize=10)

    def set_pair_plot(self, columns=None, hue="group", palette='flare', corner=True, 
                  height=2.5, aspect=1, show=False, **kwargs):
        df = self.regionprops
        
        # Filter columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            columns = [col for col in columns if col not in ["group", "label", "index", "sample", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1"]]
        
        # Create pairplot with styling parameters
        self.pair_plot = sns.pairplot(
            data=df, 
            vars=columns, 
            hue=hue, 
            palette=palette, 
            corner=corner, 
            height=height, 
            aspect=aspect,
            **kwargs
        )
        
        # Apply additional styling
        self.style_pair_plot()
        
        if show:
            plt.show()
    def get_pair_plot(self):
        return self.pair_plot
    def style_pair_plot(self, title=None, xlabel=None, ylabel=None):
        """Style the existing pair plot. Most styling must be done during creation."""
        if self.pair_plot is not None:
            # Set figure size
            self.pair_plot.figure.set_size_inches(self.fig_size)
            
            # Set title
            if title:
                self.pair_plot.figure.suptitle(title, fontsize=14)
                
            # Set axis labels (this sets labels for all subplots)
            if xlabel or ylabel:
                self.pair_plot.set_axis_labels(
                    xlabel if xlabel else "X", 
                    ylabel if ylabel else "Y",
                    fontsize=10
                )
            
            # Adjust tick label sizes
            for ax in self.pair_plot.axes.flatten():
                if ax is not None:
                    plt.setp(ax.get_xticklabels(), fontsize=8)
                    plt.setp(ax.get_yticklabels(), fontsize=8)
            
            # Apply tight layout
            self.pair_plot.figure.tight_layout()
    
    
    # def plot_jointplot(self, x, y, hue='group', kind='joint', show=False, title=None, xlabel=None, ylabel=None, **kwargs):
    #     sns.set_theme(style="whitegrid")
    #     fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    #     if kind == 'scatter':
    #         sns.scatterplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
    #     elif kind == 'box':
    #         sns.boxplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
    #     elif kind == 'joint':
    #         sns.jointplot(data=self.regionprops, x=x, y=y, hue=hue, ax=ax, **kwargs)
    #     #ax.set_title(f'Population Plot: {y} vs {x}')
    #     ax.legend(loc='best', fontsize='small')
    #     if title is not None:
    #         ax.set_title(title)
    #     if xlabel is not None:
    #         ax.set_xlabel(xlabel)
    #     if ylabel is not None:
    #         ax.set_ylabel(ylabel)
    #     #plt.xticks(rotation=45, fontsize=8)
    #     #plt.yticks(fontsize=8)
    #     self.scatter_plot = fig
    #     if show:
    #         plt.show()

    def set_categorical_plot(self, kind, metric, show=False, title=None, xlabel=None, ylabel=None, annotate=True, **kwargs):
        """REF:
        seaborn.catplot(data=None, *, x=None, y=None, hue=None, row=None, col=None, kind='strip', estimator='mean',
        errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, order=None, hue_order=None, row_order=None, 
        col_order=None, col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False, formatter=None, orient=None, 
        color=None, palette=None, hue_norm=None, legend='auto', legend_out=True, sharex=True, sharey=True, margin_titles=False, 
        facet_kws=None, ci=<deprecated>, **kwargs)
        """
        sns.set_theme(style="whitegrid", palette="flare")
        
        # Plot types that only accept x or y, not both
        one_axis_plots = {"count", "bar", "point"}
        if kind in one_axis_plots:
            self.categorical_plot = sns.catplot(data=self.regionprops, x="group", kind=kind, **kwargs)
        else:
            self.categorical_plot = sns.catplot(data=self.regionprops, x="group", y=metric, kind=kind, **kwargs)
        
        if show:
            plt.show()
       
    def get_categorical_plot(self):
        return self.categorical_plot

    def style_categorical_plot(self, metric, annotate=True, title=None, xlabel=None, ylabel=None, hue=None, row=None, col=None, kind='strip', estimator='mean',
        errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, order=None, hue_order=None, row_order=None,
        col_order=None, col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False, formatter=None, orient=None,
        color=None, palette='flare', hue_norm=None, legend='auto', legend_out=True, sharex=True, sharey=True, margin_titles=False,
        facet_kws=None, **kwargs):
                
        if self.categorical_plot is None:
            print("No categorical plot available. Call set_categorical_plot() first.")
            return
        
        # Use FacetGrid methods for styling
        if title is not None:
            self.categorical_plot.figure.suptitle(title, fontsize=14)
        
        if xlabel is not None or ylabel is not None:
            self.categorical_plot.set_axis_labels(
                xlabel if xlabel else "Group", 
                ylabel if ylabel else metric
            )
        
        # Set tick parameters using FacetGrid
        self.categorical_plot.set_xticklabels(rotation=45, fontsize=8)
        self.categorical_plot.set_yticklabels(fontsize=8)
        
        # Apply tight layout
        self.categorical_plot.figure.tight_layout()
        
        if annotate:
            try:
                from statannotations.Annotator import Annotator
                
                # Get the first (and likely only) axis from the FacetGrid
                ax = self.categorical_plot.axes.flat[0]
                
                # Prepare pairs for annotation (all pairwise group comparisons)
                groups = self.regionprops['group'].unique()
                pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]

                # Get statistics from the experiment
                self.experiment.statistics.compare_groups(metric=metric)
                
                if hasattr(self.experiment.statistics, "tukey_results") and not self.experiment.statistics.tukey_results.empty:
                    pvalues = []
                    for g1, g2 in pairs:
                        match = self.experiment.statistics.tukey_results[
                            ((self.experiment.statistics.tukey_results['group1'] == g1) & (self.experiment.statistics.tukey_results['group2'] == g2)) |
                            ((self.experiment.statistics.tukey_results['group1'] == g2) & (self.experiment.statistics.tukey_results['group2'] == g1))
                        ]
                        if not match.empty:
                            pvalues.append(float(match['p-adj'].values[0]))
                        else:
                            pvalues.append(1.0)
                    
                    try:
                        annotator = Annotator(ax, pairs, data=self.regionprops, x="group", y=metric)
                        annotator.set_pvalues_and_annotate(pvalues, test_short_name="Tukey", text_format="star", loc="inside", verbose=0)
                    except Exception as e:
                        print(f"Could not add annotations: {e}")
                        
            except ImportError:
                print("statannotations not available. Skipping annotations.")
        
        # Set final title if none provided
        if title is None:
            self.categorical_plot.figure.suptitle(f'{metric.title()} Comparison by Group', fontsize=14)
        
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

        #pca = PCA(n_components=n_components)
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
        if self.joint_plot:
            filename = asksaveasfilename(title="Save Joint Plot", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                self.joint_plot.savefig(filename)
        if self.pair_plot:
            filename = asksaveasfilename(title="Save Pair Plot", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                self.pair_plot.savefig(filename)
        if self.categorical_plot:
            filename = asksaveasfilename(title="Save Categorical Plot", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                self.categorical_plot.savefig(filename)

def test_plotting():

    # Generate some test data
    test_data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4],
        'value2': [20, 25, 26, 27]
    })

    # Create a mock ExperimentPlots instance
    plots = ExperimentPlots(regionprops=test_data)

    # Test joint plot - remove show from kwargs
    plots.set_joint_plot(x='value', y='value2', kind='kde', show=True)
    assert plots.joint_plot is not None

    # Test pair plot
    plots.set_pair_plot(show=True)
    assert plots.pair_plot is not None

    # Test categorical plot - fix the method call
    plots.set_categorical_plot(kind='violin', metric='value', show=True, annotate=True)
    assert plots.categorical_plot is not None

if __name__ == "__main__":
    test_plotting()