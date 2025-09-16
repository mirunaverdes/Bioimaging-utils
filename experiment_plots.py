import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tkinter.filedialog import asksaveasfilename
from typing import TYPE_CHECKING, Optional, List, Union, Tuple
import warnings

if TYPE_CHECKING:
    from experiment import Experiment

class ExperimentPlots:
    """
    A comprehensive plotting class for experiment data analysis.
    Provides consistent styling and easy-to-use methods for various plot types.
    """
    
    def __init__(self, experiment: "Experiment" = None, regionprops: pd.DataFrame = None, 
                 fig_size: Tuple[int, int] = (10, 6), style: str = "whitegrid", 
                 palette: str = "Set2", context: str = "notebook"):
        """
        Initialize the plotting class.
        
        Args:
            experiment: Experiment object containing data and statistics
            regionprops: DataFrame with region properties data
            fig_size: Default figure size (width, height)
            style: Seaborn style theme
            palette: Default color palette
            context: Seaborn context for scaling
        """
        self.experiment: "Experiment" = experiment
        self.regionprops = regionprops if regionprops is not None else (
            experiment.regionprops if experiment else pd.DataFrame()
        )
        self.fig_size = fig_size
        self.style = style
        self.palette = palette
        self.context = context
        
        # Plot storage
        self.joint_plot: Optional[Figure] = None
        self.pair_plot: Optional[Figure] = None
        self.categorical_plot: Optional[Figure] = None
        self.pca_plot: Optional[Figure] = None
        
        # Set default theme
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply consistent theming to all plots"""
        sns.set_theme(style=self.style, palette=self.palette, context=self.context)
        plt.rcParams.update({
            'figure.figsize': self.fig_size,
            'figure.dpi': 100,  # Good balance for screen display
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'font.size': 9,
            'figure.constrained_layout.use': True  # Better automatic layout
        })
    
    def _get_numeric_columns(self, exclude_cols: List[str] = None) -> List[str]:
        """Get numeric columns excluding specified ones"""
        if exclude_cols is None:
            exclude_cols = ["group", "label", "index", "sample", "frame", "scene",
                          "bbox-0", "bbox-1", "bbox-2", "bbox-3", 
                          "centroid-0", "centroid-1", "track_id"]
        
        numeric_cols = self.regionprops.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def plot_jointplot(self, x: str, y: str, hue: str = 'group', kind: str = 'scatter',
                      title: str = None, xlabel: str = None, ylabel: str = None,
                      show: bool = False, save_path: str = None, **kwargs) -> Figure:
        """
        Create a joint plot with marginal distributions.
        
        Args:
            x, y: Column names for x and y axes
            hue: Column name for color grouping
            kind: Type of plot ('scatter', 'kde', 'hex', 'reg')
            title, xlabel, ylabel: Plot labels
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments for sns.jointplot
            
        Returns:
            matplotlib Figure object
        """
        # Set default labels
        xlabel = xlabel or x.replace('_', ' ').title()
        ylabel = ylabel or y.replace('_', ' ').title()
        title = title or f'{ylabel} vs {xlabel}'
        
        # Create joint plot
        grid = sns.jointplot(
            data=self.regionprops, x=x, y=y, hue=hue, kind=kind,
            height=self.fig_size[1], **kwargs
        )
        
        # Style the plot
        grid.figure.suptitle(title, fontsize=14, y=1.02)
        grid.ax_joint.set_xlabel(xlabel, fontsize=12)
        grid.ax_joint.set_ylabel(ylabel, fontsize=12)
        
        # Adjust layout
        grid.figure.tight_layout()
        
        # Store the figure
        self.joint_plot = grid.figure
        
        if save_path:
            self.joint_plot.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return self.joint_plot
    
    def plot_pairplot(self, columns: List[str] = None, hue: str = "group", 
                     title: str = None, corner: bool = True, height: float = 2.5,
                     show: bool = False, save_path: str = None, **kwargs) -> Figure:
        """
        Create a pair plot matrix.
        
        Args:
            columns: List of columns to include
            hue: Column for color grouping
            title: Plot title
            corner: Whether to show only lower triangle
            height: Height of each subplot
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments for sns.pairplot
            
        Returns:
            matplotlib Figure object
        """
        # Get columns if not specified
        if columns is None:
            columns = self._get_numeric_columns()
            # Limit to first 6 columns for readability
            columns = columns[:6]
        
        # Set default title
        title = title or "Pairwise Relationships"
        
        # Create pair plot
        grid = sns.pairplot(
            data=self.regionprops, vars=columns, hue=hue, 
            corner=corner, height=height, aspect=1,
            palette=self.palette, **kwargs
        )
        
        # Style the plot
        grid.figure.suptitle(title, fontsize=16, y=1.02)
        
        # Adjust layout
        grid.figure.tight_layout()
        
        # Store the figure
        self.pair_plot = grid.figure
        
        if save_path:
            self.pair_plot.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return self.pair_plot
    
    def plot_categorical_comparisons(self, metric: str, plot_kind: str = 'box',
                                   hue: str = None, title: str = None, 
                                   xlabel: str = None, ylabel: str = None,
                                   annotate: bool = True, show: bool = False,
                                   save_path: str = None, **kwargs) -> Figure:
        """
        Create categorical comparison plots.
        
        Args:
            metric: Column name for the metric to compare
            plot_kind: Type of plot ('box', 'violin', 'strip', 'swarm', 'bar', 'point')
            hue: Additional grouping variable
            title, xlabel, ylabel: Plot labels
            annotate: Whether to add statistical annotations
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments for sns.catplot
            
        Returns:
            matplotlib Figure object
        """
        # Set default labels
        xlabel = xlabel or 'Group'
        ylabel = ylabel or metric.replace('_', ' ').title()
        title = title or f'{ylabel} by {xlabel}'
        
        # Create categorical plot
        if plot_kind in ['count', 'bar'] and hue is None:
            # For count plots, set hue to x and disable legend to avoid warning
            grid = sns.catplot(
                data=self.regionprops, x="group", hue="group", kind=plot_kind,
                height=self.fig_size[1], aspect=self.fig_size[0]/self.fig_size[1],
                palette=self.palette, legend=False, **kwargs
            )
        else:
            grid = sns.catplot(
                data=self.regionprops, x="group", y=metric, hue=hue, kind=plot_kind,
                height=self.fig_size[1], aspect=self.fig_size[0]/self.fig_size[1],
                palette=self.palette, **kwargs
            )
        
        # Style the plot
        grid.figure.suptitle(title, fontsize=12, y=1.02)
        grid.set_axis_labels(xlabel, ylabel)
        grid.set_xticklabels(rotation=45)
        
        # Add statistical annotations if requested
        if annotate and plot_kind not in ['count', 'bar']:
            self._add_statistical_annotations(grid.axes.flat[0], metric)
        
        # Adjust layout
        grid.figure.tight_layout()
        
        # Store the figure
        self.categorical_plot = grid.figure
        
        if save_path:
            self.categorical_plot.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return self.categorical_plot
    
    def _add_statistical_annotations(self, ax, metric: str):
        """Add statistical significance annotations to plots"""
        try:
            from statannotations.Annotator import Annotator
            
            # Get unique groups
            groups = self.regionprops['group'].unique()
            if len(groups) < 2:
                return
            
            # Create pairs for comparison
            pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]
            
            # Get p-values from experiment statistics
            if (self.experiment and hasattr(self.experiment, 'statistics') and 
                hasattr(self.experiment.statistics, 'tukey_results')):
                
                pvalues = self._extract_pvalues(pairs, metric)
                
                if pvalues:
                    annotator = Annotator(ax, pairs, data=self.regionprops, 
                                        x="group", y=metric)
                    annotator.set_pvalues_and_annotate(
                        pvalues, test_short_name="Tukey", 
                        text_format="star", loc="inside", verbose=0
                    )
                    
        except ImportError:
            warnings.warn("statannotations not available. Skipping statistical annotations.")
        except Exception as e:
            warnings.warn(f"Could not add statistical annotations: {e}")
    
    def _extract_pvalues(self, pairs: List[Tuple], metric: str) -> List[float]:
        """Extract p-values for group pairs from experiment statistics"""
        pvalues = []
        tukey_results = self.experiment.statistics.tukey_results
        
        for g1, g2 in pairs:
            match = tukey_results[
                ((tukey_results['group1'] == g1) & (tukey_results['group2'] == g2)) |
                ((tukey_results['group1'] == g2) & (tukey_results['group2'] == g1))
            ]
            if not match.empty:
                pvalues.append(float(match['p-adj'].values[0]))
            else:
                pvalues.append(1.0)
        
        return pvalues
    
    def plot_pca(self, n_components: int = 2, columns: List[str] = None,
                title: str = None, show: bool = False, save_path: str = None) -> Figure:
        """
        Create PCA plot with explained variance.
        
        Args:
            n_components: Number of components to plot (2 or 3)
            columns: Columns to include in PCA
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required for PCA analysis")
        
        # Get columns for PCA
        if columns is None:
            columns = self._get_numeric_columns()
        
        # Prepare data
        X = self.regionprops[columns].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        # Create plot
        if n_components == 2:
            fig, ax = plt.subplots(figsize=self.fig_size)
            scatter = ax.scatter(components[:, 0], components[:, 1], 
                               c=self.regionprops['group'].astype('category').cat.codes,
                               cmap='Set2', alpha=0.7)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            
            # Add legend
            groups = self.regionprops['group'].unique()
            for i, group in enumerate(groups):
                ax.scatter([], [], c=f'C{i}', label=group)
            ax.legend()
            
        elif n_components == 3:
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2],
                               c=self.regionprops['group'].astype('category').cat.codes,
                               cmap='Set2', alpha=0.7)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        
        # Set title
        title = title or f'PCA Analysis ({pca.explained_variance_ratio_.sum():.1%} total variance)'
        fig.suptitle(title, fontsize=14)
        
        # Store figure
        self.pca_plot = fig
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def save_all_plots(self, directory: str = None, prefix: str = "plot"):
        """Save all generated plots to files"""
        plots = {
            'joint': self.joint_plot,
            'pair': self.pair_plot,
            'categorical': self.categorical_plot,
            'pca': self.pca_plot
        }
        
        for plot_type, plot_obj in plots.items():
            if plot_obj is not None:
                if directory:
                    filename = f"{directory}/{prefix}_{plot_type}.png"
                else:
                    filename = asksaveasfilename(
                        title=f"Save {plot_type.title()} Plot",
                        defaultextension=".png",
                        filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                                 ("SVG files", "*.svg"), ("All files", "*.*")]
                    )
                
                if filename:
                    plot_obj.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Saved {plot_type} plot to {filename}")
    
    def clear_plots(self):
        """Clear all stored plots to free memory"""
        plots = [self.joint_plot, self.pair_plot, self.categorical_plot, self.pca_plot]
        for plot in plots:
            if plot:
                plt.close(plot)
        
        self.joint_plot = None
        self.pair_plot = None
        self.categorical_plot = None
        self.pca_plot = None


def test_plotting():
    """Test function for the plotting class"""
    # Generate test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'group': ['Control'] * 50 + ['Treatment'] * 50,
        'area': np.random.normal(100, 20, 100),
        'perimeter': np.random.normal(50, 10, 100),
        'eccentricity': np.random.uniform(0, 1, 100),
        'solidity': np.random.uniform(0.8, 1.0, 100)
    })
    
    # Create plotting instance
    plots = ExperimentPlots(regionprops=test_data)
    
    # Test different plot types
    print("Testing joint plot...")
    plots.plot_jointplot(x='area', y='perimeter', show=True)
    
    print("Testing pair plot...")
    plots.plot_pairplot(show=True)
    
    print("Testing categorical plot...")
    plots.plot_categorical_comparisons(metric='area', show=True)
    
    print("Testing PCA plot...")
    plots.plot_pca(show=True)
    
    print("All tests completed!")


if __name__ == "__main__":
    test_plotting()