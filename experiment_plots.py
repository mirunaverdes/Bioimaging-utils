import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tkinter.filedialog import asksaveasfilename, askdirectory
from typing import TYPE_CHECKING, Optional, List, Union, Tuple
import warnings
import copy
from utils import namedir
import os
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
            # Remove the problematic constrained_layout setting
            # 'figure.constrained_layout.use': True  # Commented out to avoid colorbar conflicts
        })
    
    def _get_white_background_style(self):
        """Get style parameters for white background publication plots"""
        return {
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.0,
            'axes.labelcolor': 'black',
            'axes.titlecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'grid.color': '#E0E0E0',
            'grid.alpha': 0.8,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False
        }
    
    def style_all_legends_white(self, fig):
        """Comprehensively style all legends in a figure for white background publication"""
        legends_to_style = []
        
        # Method 1: Figure-level legends
        if hasattr(fig, '_legend') and fig._legend:
            legends_to_style.append(fig._legend)
        
        # Method 2: Figure.legends list
        if hasattr(fig, 'legends') and fig.legends:
            legends_to_style.extend(fig.legends)
        
        # Method 3: Axes-level legends
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend:
                legends_to_style.append(legend)
        
        # Method 4: Search all figure children for legend-like objects
        def find_legends_recursive(obj):
            legends = []
            if hasattr(obj, 'get_texts') and hasattr(obj, 'get_frame'):
                try:
                    # Test if it behaves like a legend
                    obj.get_frame()
                    obj.get_texts()
                    legends.append(obj)
                except:
                    pass
            
            if hasattr(obj, 'get_children'):
                for child in obj.get_children():
                    legends.extend(find_legends_recursive(child))
            
            return legends
        
        legends_to_style.extend(find_legends_recursive(fig))
        
        # Remove duplicates
        legends_to_style = list(set(legends_to_style))
        
        # Style all found legends for white background
        for legend in legends_to_style:
            try:
                frame = legend.get_frame()
                frame.set_facecolor('white')
                frame.set_edgecolor('black')
                frame.set_linewidth(0.5)
                frame.set_alpha(1.0)
                
                # Style legend text
                for text in legend.get_texts():
                    text.set_color('black')
                    text.set_fontsize(9)
                    text.set_fontweight('normal')
                
                # Style legend title if present
                if hasattr(legend, 'get_title') and legend.get_title():
                    title = legend.get_title()
                    title.set_color('black')
                    title.set_fontsize(10)
                    title.set_fontweight('bold')
                
                # Style legend markers/handles
                for handle in legend.legendHandles:
                    if hasattr(handle, 'set_edgecolor'):
                        # For scatter plots, line plots, etc.
                        handle.set_edgecolor('black')
                        handle.set_linewidth(0.5)
                    
                    if hasattr(handle, 'set_markeredgecolor'):
                        # For markers specifically
                        handle.set_markeredgecolor('black')
                        handle.set_markeredgewidth(0.5)
                
                # Set legend positioning and spacing
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc('upper left')
                
                # Adjust legend spacing
                legend.set_columnspacing(1.0)
                legend.set_handletextpad(0.5)
                legend.set_handlelength(1.5)
                legend.set_borderpad(0.5)
                
            except Exception as e:
                print(f"Could not style legend: {e}")
                continue
        
        return len(legends_to_style)  # Return number of legends styled

    def style_all_legends_dark(self, fig):
        """Comprehensively style all legends in a figure for dark background interface"""
        legends_to_style = []
        
        # Method 1: Figure-level legends
        if hasattr(fig, '_legend') and fig._legend:
            legends_to_style.append(fig._legend)
        
        # Method 2: Figure.legends list
        if hasattr(fig, 'legends') and fig.legends:
            legends_to_style.extend(fig.legends)
        
        # Method 3: Axes-level legends
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend:
                legends_to_style.append(legend)
        
        # Method 4: Search all figure children for legend-like objects
        def find_legends_recursive(obj):
            legends = []
            if hasattr(obj, 'get_texts') and hasattr(obj, 'get_frame'):
                try:
                    # Test if it behaves like a legend
                    obj.get_frame()
                    obj.get_texts()
                    legends.append(obj)
                except:
                    pass
            
            if hasattr(obj, 'get_children'):
                for child in obj.get_children():
                    legends.extend(find_legends_recursive(child))
            
            return legends
        
        legends_to_style.extend(find_legends_recursive(fig))
        
        # Remove duplicates
        legends_to_style = list(set(legends_to_style))
        
        # Style all found legends for dark background
        for legend in legends_to_style:
            try:
                frame = legend.get_frame()
                frame.set_facecolor('#2b2b2b')
                frame.set_edgecolor('#ffffff')
                frame.set_linewidth(0.5)
                frame.set_alpha(0.9)
                
                # Style legend text
                for text in legend.get_texts():
                    text.set_color('#ffffff')
                    text.set_fontsize(9)
                    text.set_fontweight('normal')
                
                # Style legend title if present
                if hasattr(legend, 'get_title') and legend.get_title():
                    title = legend.get_title()
                    title.set_color('#ffffff')
                    title.set_fontsize(10)
                    title.set_fontweight('bold')
                
                # Style legend markers/handles
                for handle in legend.legendHandles:
                    if hasattr(handle, 'set_edgecolor'):
                        # For scatter plots, line plots, etc.
                        handle.set_edgecolor('#ffffff')
                        handle.set_linewidth(0.5)
                    
                    if hasattr(handle, 'set_markeredgecolor'):
                        # For markers specifically
                        handle.set_markeredgecolor('#ffffff')
                        handle.set_markeredgewidth(0.5)
                
                # Set legend positioning and spacing
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc('upper left')
                
                # Adjust legend spacing
                legend.set_columnspacing(1.0)
                legend.set_handletextpad(0.5)
                legend.set_handlelength(1.5)
                legend.set_borderpad(0.5)
                
            except Exception as e:
                print(f"Could not style legend: {e}")
                continue
        
        return len(legends_to_style)  # Return number of legends styled

    # Update the _apply_white_background_to_figure method to use the new function:

    def _apply_white_background_to_figure(self, fig):
        """Apply white background styling to a figure for publication"""
        # Set figure background
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        
        # Style all axes
        for ax in fig.get_axes():
            # Background and edges
            ax.set_facecolor('white')
            ax.patch.set_alpha(1.0)
            
            # Spines (borders)
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # Only show left and bottom spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Tick parameters
            ax.tick_params(colors='black', which='both')
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Labels and title
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), color='black', fontsize=10)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), color='black', fontsize=10)
            if ax.get_title():
                ax.set_title(ax.get_title(), color='black', fontsize=12)
            
            # Grid
            ax.grid(True, color='#E0E0E0', alpha=0.8, linewidth=0.5)
        
        # Style all legends comprehensively for white background
        self.style_all_legends_white(fig)

    # Also add a general function that can be called directly:

    def apply_white_background_styling(self, fig, include_legends: bool = True):
        """
        Apply comprehensive white background styling to any figure.
        
        Args:
            fig: matplotlib Figure object
            include_legends: Whether to style legends as well
            
        Returns:
            Number of legends styled (if include_legends=True)
        """
        # Apply the white background styling
        self._apply_white_background_to_figure(fig)
        
        # Optionally style legends
        if include_legends:
            return self.style_all_legends_white(fig)
        
        return 0

    def apply_dark_background_styling(self, fig, include_legends: bool = True):
        """
        Apply comprehensive dark background styling to any figure.
        
        Args:
            fig: matplotlib Figure object
            include_legends: Whether to style legends as well
            
        Returns:
            Number of legends styled (if include_legends=True)
        """
        # Set figure background
        fig.patch.set_facecolor('#212121')
        fig.patch.set_alpha(1.0)
        
        # Style all axes
        for ax in fig.get_axes():
            # Background and edges
            ax.set_facecolor('#2b2b2b')
            ax.patch.set_alpha(1.0)
            
            # Spines (borders)
            for spine in ax.spines.values():
                spine.set_color('#ffffff')
                spine.set_linewidth(1.0)
            
            # Only show left and bottom spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Tick parameters
            ax.tick_params(colors='#ffffff', which='both')
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Labels and title
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), color='#ffffff', fontsize=10)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), color='#ffffff', fontsize=10)
            if ax.get_title():
                ax.set_title(ax.get_title(), color='#ffffff', fontsize=12)
            
            # Grid
            ax.grid(True, color='#ffffff', alpha=0.3, linewidth=0.5)
        
        # Style all legends comprehensively for dark background
        if include_legends:
            return self.style_all_legends_dark(fig)
        
        return 0

    def save_publication_plot(self, figure, filepath: str = None,
                               dpi: int = 300, transparent: bool = True,
                               file_formats: List[str] = ['png'], **kwargs):
        """
        Save a plot with publication-ready styling.
        
        Args:
            figure: matplotlib Figure object to save
            filename: Base filename (without extension)
            dpi: Resolution for saving
            transparent: Whether to use transparent background
            file_formats: List of formats to save ['png', 'pdf', 'svg', 'eps']
            **kwargs: Additional arguments for savefig
        """
        if figure is None:
            warnings.warn("No figure provided for saving")
            return
        
        # Get filename if not provided or invalid
        if filepath is None:
            filepath = asksaveasfilename(
                title="Save Publication Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"), 
                    ("PDF files", "*.pdf"), 
                    ("SVG files", "*.svg"),
                    ("EPS files", "*.eps"),
                    ("All files", "*.*"),
                ],
            )
            if not filepath:
                return
            
            # Remove extension to add multiple formats
            if '.' in filepath:
                filepath = '.'.join(filepath.split('.')[:-1])
        
        # Create a copy of the figure for modification
        fig_copy = copy.deepcopy(figure)
        
        # Apply white background styling to the copy
        self._apply_white_background_to_figure(fig_copy)
        
        # Default save parameters for publication quality
        save_params = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': transparent,
            'facecolor': 'white' if not transparent else 'none',
            'edgecolor': 'none'
        }
        save_params.update(kwargs)
        
        # Save in requested formats
        saved_files = []
        for fmt in file_formats:
            try:
                save_filename = f"{filepath}.{fmt}"
                
                # Format-specific adjustments
                if fmt.lower() == 'pdf':
                    save_params_fmt = save_params.copy()
                    save_params_fmt['transparent'] = False  # PDF doesn't handle transparency well
                    save_params_fmt['facecolor'] = 'white'
                elif fmt.lower() == 'eps':
                    save_params_fmt = save_params.copy()
                    save_params_fmt['transparent'] = False  # EPS doesn't handle transparency
                    save_params_fmt['facecolor'] = 'white'
                else:
                    save_params_fmt = save_params.copy()
                
                fig_copy.savefig(save_filename, format=fmt, **save_params_fmt)
                saved_files.append(save_filename)
                print(f"✅ Saved: {os.path.basename(save_filename)}")
                
            except Exception as e:
                print(f"❌ Error saving {fmt}: {e}")
        
        # Clean up the copy
        plt.close(fig_copy)
        
        return saved_files
    
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
        
        # Use subplots_adjust instead of tight_layout for jointplot
        grid.figure.subplots_adjust(top=0.9)
        
        # Store the figure
        self.joint_plot = grid.figure
        
        if save_path:
            self.save_publication_plot(self.joint_plot, save_path)
        
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
        
        # Use subplots_adjust instead of tight_layout for pairplot
        grid.figure.subplots_adjust(top=0.95)
        
        # Store the figure
        self.pair_plot = grid.figure
        
        if save_path:
            self.save_publication_plot(self.pair_plot, save_path)
        
        if show:
            plt.show()
        
        return self.pair_plot

    def plot_categorical_comparisons(self, metric: str, groupaxis: str = 'group', plot_kind: str = 'violin',
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
                data=self.regionprops, x=groupaxis, hue=groupaxis, kind=plot_kind,
                height=self.fig_size[1], aspect=self.fig_size[0]/self.fig_size[1],
                palette=self.palette, legend=False, **kwargs
            )
        else:
            grid = sns.catplot(
                data=self.regionprops, x=groupaxis, y=metric, hue=hue, kind=plot_kind,
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
        
        # Use subplots_adjust instead of tight_layout
        grid.figure.subplots_adjust(bottom=0.15)
        
        # Store the figure
        self.categorical_plot = grid.figure
        
        if save_path:
            self.save_publication_plot(self.categorical_plot, save_path)
        
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
                title: str = None, show: bool = False, save_path: str = None,
                show_loadings: bool = True, n_top_features: int = 3) -> Figure:
        """
        Create PCA plot with explained variance and feature loadings.
        
        Args:
            n_components: Number of components to plot (2 or 3)
            columns: Columns to include in PCA
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the plot
            show_loadings: Whether to print feature loadings
            n_top_features: Number of top features to display for each component
            
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
        
        # Store PCA results for later access
        self.pca_model = pca
        self.pca_features = columns
        self.pca_components = components
        
        # Extract and display feature loadings
        if show_loadings:
            self._display_pca_loadings(pca, columns, n_top_features)
        
        # Create plot
        if n_components == 2:
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            # Get unique groups and create color mapping
            groups = self.regionprops['group'].unique()
            colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
            
            # Plot each group separately for better legend
            for i, group in enumerate(groups):
                mask = self.regionprops['group'] == group
                ax.scatter(components[mask, 0], components[mask, 1], 
                        c=[colors[i]], label=group, alpha=0.7, s=50)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif n_components == 3:
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.add_subplot(111, projection='3d')
            
            # Get unique groups and create color mapping
            groups = self.regionprops['group'].unique()
            colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
            
            # Plot each group separately for better legend
            for i, group in enumerate(groups):
                mask = self.regionprops['group'] == group
                ax.scatter(components[mask, 0], components[mask, 1], components[mask, 2],
                        c=[colors[i]], label=group, alpha=0.7, s=50)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.legend()
        
        # Set title
        title = title or f'PCA Analysis ({pca.explained_variance_ratio_.sum():.1%} total variance)'
        fig.suptitle(title, fontsize=14)
        
        # Store figure
        self.pca_plot = fig
        
        if save_path:
            self.save_publication_plot(self.pca_plot, save_path)
        
        if show:
            plt.show()
        
        return fig

    def _display_pca_loadings(self, pca, feature_names: List[str], n_top: int = 3):
        """
        Display PCA loadings (component weights) for each principal component.
        
        Args:
            pca: Fitted PCA model
            feature_names: List of feature names
            n_top: Number of top features to display for each component
        """
        print("\n" + "="*60)
        print("🔍 PCA COMPONENT LOADINGS (Feature Importance)")
        print("="*60)
        
        # Get loadings matrix (components_ is features x components)
        loadings = pca.components_.T  # Transpose to get features x components
        
        for i in range(pca.n_components_):
            print(f"\n📊 PC{i+1} ({pca.explained_variance_ratio_[i]:.1%} variance):")
            print("-" * 40)
            
            # Get absolute loadings for ranking
            component_loadings = loadings[:, i]
            abs_loadings = np.abs(component_loadings)
            
            # Get top features
            top_indices = np.argsort(abs_loadings)[::-1][:n_top]
            
            print("Top contributing features:")
            for rank, idx in enumerate(top_indices, 1):
                feature = feature_names[idx]
                loading = component_loadings[idx]
                abs_loading = abs_loadings[idx]
                direction = "+" if loading > 0 else "-"
                
                print(f"  {rank}. {feature:<20} | {direction}{abs_loading:.3f} | {loading:+.3f}")
            
            # Show all features if requested
            if len(feature_names) <= 10:  # Only show all if not too many
                print(f"\nAll features for PC{i+1}:")
                sorted_indices = np.argsort(abs_loadings)[::-1]
                for idx in sorted_indices:
                    feature = feature_names[idx]
                    loading = component_loadings[idx]
                    direction = "+" if loading > 0 else "-"
                    print(f"  {feature:<20} | {direction}{abs(loading):.3f}")

    def get_pca_loadings_dataframe(self) -> pd.DataFrame:
        """
        Get PCA loadings as a pandas DataFrame for further analysis.
        
        Returns:
            DataFrame with features as rows and components as columns
        """
        if not hasattr(self, 'pca_model') or self.pca_model is None:
            raise ValueError("PCA has not been performed yet. Run plot_pca() first.")
        
        # Create DataFrame with loadings
        loadings_df = pd.DataFrame(
            self.pca_model.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca_model.n_components_)],
            index=self.pca_features
        )
        
        # Add absolute values for easier sorting
        for i in range(self.pca_model.n_components_):
            loadings_df[f'PC{i+1}_abs'] = np.abs(loadings_df[f'PC{i+1}'])
        
        return loadings_df

    def plot_pca_loadings(self, component: int = 1, n_features: int = 10, 
                        save_path: str = None) -> Figure:
        """
        Create a bar plot of PCA loadings for a specific component.
        
        Args:
            component: Which component to plot (1-indexed)
            n_features: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if not hasattr(self, 'pca_model') or self.pca_model is None:
            raise ValueError("PCA has not been performed yet. Run plot_pca() first.")
        
        if component > self.pca_model.n_components_:
            raise ValueError(f"Component {component} not available. Only {self.pca_model.n_components_} components computed.")
        
        # Get loadings for specified component (convert to 0-indexed)
        comp_idx = component - 1
        loadings = self.pca_model.components_[comp_idx]
        
        # Get top features by absolute loading
        abs_loadings = np.abs(loadings)
        top_indices = np.argsort(abs_loadings)[::-1][:n_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = [self.pca_features[i] for i in top_indices]
        top_loadings = [loadings[i] for i in top_indices]
        
        # Create bar plot
        colors = ['red' if x < 0 else 'blue' for x in top_loadings]
        bars = ax.barh(range(len(top_features)), top_loadings, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features])
        ax.set_xlabel(f'PC{component} Loading')
        ax.set_title(f'PC{component} Feature Loadings ({self.pca_model.explained_variance_ratio_[comp_idx]:.1%} variance)')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        # Add value labels on bars
        for i, (bar, loading) in enumerate(zip(bars, top_loadings)):
            ax.text(loading + (0.02 if loading > 0 else -0.02), i, f'{loading:.3f}', 
                    va='center', ha='left' if loading > 0 else 'right', fontsize=9)
        
        # Invert y-axis to show highest loading at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Store figure
        self.pca_loadings_plot = fig
        
        if save_path:
            self.save_publication_plot(fig, save_path)
        
        return fig

    def get_top_pca_features(self, component: int = 1, n_features: int = 3) -> pd.DataFrame:
        """
        Get the top features for a specific PCA component.
        
        Args:
            component: Which component to analyze (1-indexed)
            n_features: Number of top features to return
            
        Returns:
            DataFrame with feature names, loadings, and absolute loadings
        """
        if not hasattr(self, 'pca_model') or self.pca_model is None:
            raise ValueError("PCA has not been performed yet. Run plot_pca() first.")
        
        if component > self.pca_model.n_components_:
            raise ValueError(f"Component {component} not available. Only {self.pca_model.n_components_} components computed.")
        
        # Get loadings for specified component (convert to 0-indexed)
        comp_idx = component - 1
        loadings = self.pca_model.components_[comp_idx]
        
        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': self.pca_features,
            'loading': loadings,
            'abs_loading': np.abs(loadings)
        })
        
        # Sort by absolute loading and get top features
        top_features = feature_df.nlargest(n_features, 'abs_loading')
        
        return top_features

    # Add this method to make it easy to use from the app
    def analyze_pca_components(self, n_components: int = 2, n_top_features: int = 3,
                            show_plots: bool = True) -> dict:
        """
        Comprehensive PCA analysis with loadings and top features.
        
        Args:
            n_components: Number of components to analyze
            n_top_features: Number of top features to show for each component
            show_plots: Whether to create plots
            
        Returns:
            Dictionary with PCA results and top features
        """
        # Run PCA
        pca_fig = self.plot_pca(n_components=n_components, show_loadings=True, 
                            n_top_features=n_top_features, show=show_plots)
        
        results = {
            'pca_figure': pca_fig,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'total_variance_explained': self.pca_model.explained_variance_ratio_.sum(),
            'top_features_by_component': {}
        }
        
        # Get top features for each component
        for i in range(n_components):
            component_num = i + 1
            top_features = self.get_top_pca_features(component_num, n_top_features)
            results['top_features_by_component'][f'PC{component_num}'] = top_features
            
            if show_plots and i == 0:  # Show loadings plot for first component
                loadings_fig = self.plot_pca_loadings(component_num, n_features=min(10, len(self.pca_features)))
                results['loadings_figure'] = loadings_fig
        
        return results
    
    def save_all_plots(self, directory: str = None, prefix: str = "plot",
                      file_formats: List[str] = ['png'], **kwargs):
        """
        Save all generated plots to files with publication quality.
        
        Args:
            directory: Directory to save plots (if None, will prompt for each)
            prefix: Prefix for filenames
            file_formats: List of formats to save ['png', 'pdf', 'svg', 'eps']
            **kwargs: Additional arguments for save_publication_plot
        """
        plots = {
            'joint': self.joint_plot,
            'pair': self.pair_plot,
            'categorical': self.categorical_plot,
            'pca': self.pca_plot,
            'directionality': getattr(self, 'directionality_plot', None),
            'directionality_rose': getattr(self, 'directionality_rose_plot', None),
            'directionality_comparison': getattr(self, 'directionality_comparison_plot', None),
            'time_series': getattr(self, 'time_series_plot', None),
            'multi_metric_time_series': getattr(self, 'multi_metric_time_series_plot', None),
            'time_series_heatmap': getattr(self, 'time_series_heatmap', None),
            'temporal_trends': getattr(self, 'temporal_trends_plot', None)
        }
        
        saved_files = []
        for plot_type, plot_obj in plots.items():
            if plot_obj is not None:
                if directory:
                    filename = f"{directory}/{prefix}_{plot_type}"
                else:
                    filename = None  # Will prompt user
                
                files = self.save_publication_plot(
                    plot_obj, filename, file_formats=file_formats, **kwargs
                )
                if files:
                    saved_files.extend(files)
        
        return saved_files
    
    def clear_plots(self):
        """Clear all stored plots to free memory"""
        plots = [
            self.joint_plot, self.pair_plot, self.categorical_plot, self.pca_plot,
            getattr(self, 'directionality_plot', None),
            getattr(self, 'directionality_rose_plot', None),
            getattr(self, 'directionality_comparison_plot', None),
            getattr(self, 'time_series_plot', None),
            getattr(self, 'multi_metric_time_series_plot', None),
            getattr(self, 'time_series_heatmap', None),
            getattr(self, 'temporal_trends_plot', None)
        ]
        for plot in plots:
            if plot:
                plt.close(plot)
        
        self.joint_plot = None
        self.pair_plot = None
        self.categorical_plot = None
        self.pca_plot = None
        self.directionality_plot = None
        self.directionality_rose_plot = None
        self.directionality_comparison_plot = None
        self.time_series_plot = None
        self.multi_metric_time_series_plot = None
        self.time_series_heatmap = None
        self.temporal_trends_plot = None

    def plot_directionality(self, directionality_col: str = 'orientation', 
                       group_col: str = 'group', title: str = None,
                       bins: int = 12, stat: str = 'density',
                       show: bool = False, save_path: str = None,
                       figsize: Tuple[int, int] = None, **kwargs) -> Figure:
        """
        Create polar histogram plots showing directionality/orientation distributions for each group.
        
        Args:
            directionality_col: Column name containing directionality/orientation data (in radians)
            group_col: Column name for grouping (default 'group')
            title: Overall plot title
            bins: Number of bins for histogram
            stat: Statistic to compute ('density', 'count', 'probability')
            show: Whether to display the plot
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            **kwargs: Additional arguments for sns.histplot
            
        Returns:
            matplotlib Figure object
        """
        # Check if directionality column exists
        if directionality_col not in self.regionprops.columns:
            available_cols = [col for col in self.regionprops.columns if 
                             any(term in col.lower() for term in ['orientation', 'angle', 'direction', 'theta'])]
            if available_cols:
                print(f"Warning: '{directionality_col}' not found. Available orientation columns: {available_cols}")
                if 'orientation' in available_cols:
                    directionality_col = 'orientation'
                    print(f"Using '{directionality_col}' instead.")
                else:
                    raise ValueError(f"Column '{directionality_col}' not found in data")
            else:
                raise ValueError(f"No directionality/orientation data found. Available columns: {list(self.regionprops.columns)}")
        
        # Get unique groups
        groups = self.regionprops[group_col].unique()
        n_groups = len(groups)
        
        # Set figure size
        if figsize is None:
            figsize = (6 * n_groups, 6)
        
        # Create figure with polar subplots
        if n_groups == 1:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=n_groups, figsize=figsize, 
                                    subplot_kw={'projection': 'polar'})
            if n_groups == 1:
                axes = [axes]
        
        # Get colors for each group
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
        
        # Plot histogram for each group
        for i, group in enumerate(groups):
            ax = axes[i]
            subset = self.regionprops[self.regionprops[group_col] == group]
            
            # Convert orientation data to appropriate range if needed
            angles = subset[directionality_col].dropna()
            
            # Ensure angles are in [0, 2π] range
            if angles.max() <= np.pi:
                # Data is in [-π/2, π/2] or [0, π], convert to [0, 2π]
                angles = angles + np.pi
            
            # Create histogram data
            hist, bin_edges = np.histogram(angles, bins=bins, density=(stat=='density'))
            
            # Calculate bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Create bar plot in polar coordinates
            if stat == 'density':
                heights = hist
            elif stat == 'count':
                heights = hist * len(angles) if stat == 'density' else hist
            elif stat == 'probability':
                heights = hist / np.sum(hist)
            else:
                heights = hist
            
            # Plot bars
            bars = ax.bar(bin_centers, heights, width=2*np.pi/bins, 
                         color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Customize polar plot
            ax.set_title(f'{group}', fontsize=14, pad=20)
            ax.set_theta_zero_location('E')  # 0° at East (right)
            ax.set_theta_direction(1)        # Counterclockwise
            
            # Set radial axis properties
            if stat == 'density':
                ax.set_ylabel('Density', fontsize=12, labelpad=30)
            elif stat == 'count':
                ax.set_ylabel('Count', fontsize=12, labelpad=30)
            elif stat == 'probability':
                ax.set_ylabel('Probability', fontsize=12, labelpad=30)
            
            ax.yaxis.set_label_position("left")
            ax.yaxis.label.set_rotation(0)
            
            # Set tick properties
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10, colors='#5c5a5a')
            
            # Set radial ticks
            max_height = np.max(heights)
            if max_height > 0:
                ax.set_rticks(np.linspace(0, max_height, 4))
            
            # Add angular grid lines
            ax.set_thetagrids(np.arange(0, 360, 45))
            ax.grid(True, alpha=0.3)
        
        # Set overall title
        if title is None:
            title = f'{directionality_col.replace("_", " ").title()} Distribution by {group_col.title()}'
        
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Use subplots_adjust instead of tight_layout for polar plots
        fig.subplots_adjust(top=0.9)
        
        # Store the figure
        self.directionality_plot = fig
        
        if save_path:
            self.save_publication_plot(self.directionality_plot, save_path)
        
        if show:
            plt.show()
        
        return fig

    def plot_directionality_rose(self, directionality_col: str = 'orientation',
                            group_col: str = 'group', title: str = None,
                            bins: int = 16, show: bool = False, 
                            save_path: str = None, **kwargs) -> Figure:
        """
        Create rose plots (wind rose style) for directionality data.
        
        Args:
            directionality_col: Column name containing directionality/orientation data
            group_col: Column name for grouping
            title: Plot title
            bins: Number of angular bins
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional styling arguments
        
        Returns:
            matplotlib Figure object
        """
        # Check if directionality column exists
        if directionality_col not in self.regionprops.columns:
            raise ValueError(f"Column '{directionality_col}' not found in data")
        
        groups = self.regionprops[group_col].unique()
        n_groups = len(groups)
        
        # Create figure
        fig, axes = plt.subplots(nrows=1, ncols=n_groups, figsize=(6*n_groups, 6),
                                subplot_kw={'projection': 'polar'})
        if n_groups == 1:
            axes = [axes]
        
        # Get colors
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
        
        for i, group in enumerate(groups):
            ax = axes[i]
            subset = self.regionprops[self.regionprops[group_col] == group]
            angles = subset[directionality_col].dropna()
            
            # Create rose plot
            theta = np.linspace(0, 2*np.pi, bins+1)
            radii, _ = np.histogram(angles, bins=theta)
            
            # Normalize to get frequencies
            radii = radii / len(angles)
            
            # Create the rose plot
            bars = ax.bar(theta[:-1], radii, width=2*np.pi/bins, 
                         bottom=0, color=colors[i], alpha=0.7, edgecolor='black')
            
            # Customize
            ax.set_title(f'{group}', fontsize=14, pad=20)
            ax.set_theta_zero_location('N')  # 0° at North (top)
            ax.set_theta_direction(-1)       # Clockwise
            ax.set_ylim(0, np.max(radii) * 1.1)
            
            # Add percentage labels
            for bar, radius in zip(bars, radii):
                if radius > 0.01:  # Only label significant bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f'{radius:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Set title
        if title is None:
            title = f'{directionality_col.replace("_", " ").title()} Rose Plot'
        fig.suptitle(title, fontsize=16)
        
        # Use subplots_adjust instead of tight_layout
        fig.subplots_adjust(top=0.9)
        
        # Store figure
        self.directionality_rose_plot = fig
        
        if save_path:
            self.save_publication_plot(self.directionality_rose_plot, save_path)
        
        if show:
            plt.show()
        
        return fig

    def plot_directionality_comparison(self, directionality_col: str = 'orientation',
                                  group_col: str = 'group', plot_type: str = 'histogram',
                                  title: str = None, show: bool = False,
                                  save_path: str = None, **kwargs) -> Figure:
        """
        Create comparison plots for directionality data across groups.
        
        Args:
            directionality_col: Column name containing directionality data
            group_col: Column name for grouping
            plot_type: Type of plot ('histogram', 'kde', 'violin', 'box')
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments
        
        Returns:
            matplotlib Figure object
        """
        if directionality_col not in self.regionprops.columns:
            raise ValueError(f"Column '{directionality_col}' not found in data")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Overlaid distributions
        groups = self.regionprops[group_col].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        
        for i, group in enumerate(groups):
            subset = self.regionprops[self.regionprops[group_col] == group]
            angles = subset[directionality_col].dropna()
            
            if plot_type == 'histogram':
                ax1.hist(angles, bins=20, alpha=0.7, label=group, 
                        color=colors[i], density=True)
            elif plot_type == 'kde':
                sns.kdeplot(data=angles, ax=ax1, label=group, color=colors[i])
        
        ax1.set_xlabel(f'{directionality_col.replace("_", " ").title()}')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Statistical comparison
        if plot_type in ['violin', 'box']:
            if plot_type == 'violin':
                sns.violinplot(data=self.regionprops, x=group_col, y=directionality_col, ax=ax2)
            else:
                sns.boxplot(data=self.regionprops, x=group_col, y=directionality_col, ax=ax2)
        else:
            # Circular statistics if available
            try:
                from scipy import stats
                # Convert to circular statistics
                groups_data = []
                for group in groups:
                    subset = self.regionprops[self.regionprops[group_col] == group]
                    angles = subset[directionality_col].dropna()
                    groups_data.append(angles)
                
                # Create violin plot as fallback
                sns.violinplot(data=self.regionprops, x=group_col, y=directionality_col, ax=ax2)
            except ImportError:
                sns.violinplot(data=self.regionprops, x=group_col, y=directionality_col, ax=ax2)
    
        ax2.set_title('Statistical Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Set overall title
        if title is None:
            title = f'{directionality_col.replace("_", " ").title()} Analysis'
        fig.suptitle(title, fontsize=16)
        
        # Use subplots_adjust instead of tight_layout
        fig.subplots_adjust(bottom=0.15)
        
        # Store figure
        self.directionality_comparison_plot = fig
        
        if save_path:
            self.save_publication_plot(self.directionality_comparison_plot, save_path)
        
        if show:
            plt.show()
        
        return fig

    def plot_time_series(self, x_col: str = 'frame', y_cols: Union[str, List[str]] = None,
                    group_col: str = 'group', sample_col: str = 'sample',
                    title: str = None, xlabel: str = None, ylabel: str = None,
                    aggregation: str = 'mean', error_bars: str = 'std',
                    show_individual: bool = False, smooth: bool = False,
                    smooth_window: int = 3, show: bool = False, 
                    save_path: str = None, figsize: Tuple[int, int] = None,
                    **kwargs) -> Figure:
        """
        Create time series line plots showing how metrics change over time.
        
        Args:
            x_col: Column name for x-axis (time variable, e.g., 'frame', 'time')
            y_cols: Column name(s) for y-axis metrics (str for single, list for multiple)
            group_col: Column name for grouping (default 'group')
            sample_col: Column name for sample identification
            title: Plot title
            xlabel, ylabel: Axis labels
            aggregation: How to aggregate data ('mean', 'median', 'sum')
            error_bars: Type of error bars ('std', 'sem', 'ci', 'none')
            show_individual: Whether to show individual sample traces
            smooth: Whether to apply smoothing
            smooth_window: Window size for smoothing
            show: Whether to display the plot
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            **kwargs: Additional arguments for plotting
            
        Returns:
            matplotlib Figure object
        """
        # Validate inputs
        if x_col not in self.regionprops.columns:
            raise ValueError(f"Time column '{x_col}' not found in data")
        
        # Handle y_cols input
        if y_cols is None:
            y_cols = self._get_numeric_columns()[:3]  # Default to first 3 numeric columns
        elif isinstance(y_cols, str):
            y_cols = [y_cols]
        
        # Validate y_cols
        missing_cols = [col for col in y_cols if col not in self.regionprops.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        # Set figure size
        if figsize is None:
            figsize = (12, 6) if len(y_cols) == 1 else (12, 4 * len(y_cols))
        
        # Create subplots
        n_plots = len(y_cols)
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=figsize, sharex=True)
            if n_plots == 1:
                axes = [axes]

        # Get unique groups
        groups = self.regionprops[group_col].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        
        # Plot each metric
        for plot_idx, y_col in enumerate(y_cols):
            ax = axes[plot_idx]
            
            # Plot each group
            for group_idx, group in enumerate(groups):
                group_data = self.regionprops[self.regionprops[group_col] == group]
                
                if group_data.empty:
                    continue
                
                # Show individual sample traces if requested
                if show_individual and sample_col in group_data.columns:
                    samples = group_data[sample_col].unique()
                    for sample in samples:
                        sample_data = group_data[group_data[sample_col] == sample]
                        if not sample_data.empty:
                            # Sort by time
                            sample_data = sample_data.sort_values(x_col)
                            ax.plot(sample_data[x_col], sample_data[y_col], 
                                color=colors[group_idx], alpha=0.3, linewidth=0.5)
                
                # Aggregate data by time point
                if aggregation == 'mean':
                    agg_data = group_data.groupby(x_col)[y_col].agg(['mean', 'std', 'sem', 'count']).reset_index()
                    y_values = agg_data['mean']
                elif aggregation == 'median':
                    agg_data = group_data.groupby(x_col)[y_col].agg(['median', 'std', 'sem', 'count']).reset_index()
                    y_values = agg_data['median']
                elif aggregation == 'sum':
                    agg_data = group_data.groupby(x_col)[y_col].agg(['sum', 'std', 'sem', 'count']).reset_index()
                    y_values = agg_data['sum']
        
                x_values = agg_data[x_col]
                
                # Apply smoothing if requested
                if smooth and len(x_values) > smooth_window:
                    from scipy.signal import savgol_filter
                    try:
                        y_values = savgol_filter(y_values, smooth_window, 2)
                    except:
                        # Fallback to simple moving average
                        y_values = y_values.rolling(window=smooth_window, center=True).mean().fillna(y_values)
                
                # Plot main line
                line = ax.plot(x_values, y_values, color=colors[group_idx], 
                            label=group, linewidth=2, marker='o', markersize=4)[0]
                
                # Add error bars
                if error_bars != 'none' and error_bars in agg_data.columns:
                    if error_bars == 'ci':
                        # Calculate 95% confidence interval
                        import scipy.stats as stats
                        error_values = agg_data['sem'] * stats.t.ppf(0.975, agg_data['count'] - 1)
                    else:
                        error_values = agg_data[error_bars]
                    
                    ax.fill_between(x_values, y_values - error_values, y_values + error_values,
                                color=colors[group_idx], alpha=0.2)
        
            # Customize subplot - MOVED OUTSIDE THE GROUP LOOP
            metric_label = ylabel if ylabel and len(y_cols) == 1 else y_col.replace('_', ' ').title()
            ax.set_ylabel(metric_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        # Set xlabel on the bottom subplot
        ax = axes[-1]
        time_label = xlabel if xlabel else x_col.replace('_', ' ').title()
        ax.set_xlabel(time_label, fontsize=12)

        # Set overall title
        if title is None:
            if len(y_cols) == 1:
                title = f'{y_cols[0].replace("_", " ").title()} Over Time'
            else:
                title = 'Time Series Analysis'
        
        fig.suptitle(title, fontsize=16, y=0.95)
        
        # Use subplots_adjust instead of tight_layout
        fig.subplots_adjust(bottom=0.1, top=0.9)
        
        # Store the figure
        self.time_series_plot = fig
        
        if save_path:
            self.save_publication_plot(self.time_series_plot, save_path)
        
        if show:
            plt.show()
        
        return fig

    def plot_temporal_trends(self, x_col: str = 'frame', y_cols: List[str] = None,
                        group_col: str = 'group', trend_analysis: bool = True,
                        show_trends: bool = True, show: bool = False,
                        save_path: str = None, **kwargs) -> Figure:
        """
        Create time series plots with trend analysis (slopes, correlations).
        
        Args:
            x_col: Column name for x-axis (time variable)
            y_cols: List of column names for metrics
            group_col: Column name for grouping
            trend_analysis: Whether to perform trend analysis
            show_trends: Whether to show trend lines
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments
        
        Returns:
            matplotlib Figure object
        """
        if y_cols is None:
            y_cols = self._get_numeric_columns()[:2]  # Default to first 2 metrics
        
        n_metrics = len(y_cols)
        fig, axes = plt.subplots(nrows=1, ncols=n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        groups = self.regionprops[group_col].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        
        trend_results = {}
        
        for metric_idx, y_col in enumerate(y_cols):
            ax = axes[metric_idx]
            
            for group_idx, group in enumerate(groups):
                group_data = self.regionprops[self.regionprops[group_col] == group]
                
                # Aggregate data
                agg_data = group_data.groupby(x_col)[y_col].mean().reset_index()
                x_values = agg_data[x_col].values
                y_values = agg_data[y_col].values
                
                # Plot data points
                ax.scatter(x_values, y_values, color=colors[group_idx], 
                        label=group, alpha=0.7, s=50)
                
                # Fit trend line if requested
                if show_trends and len(x_values) > 2:
                    try:
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                        
                        # Plot trend line
                        trend_line = slope * x_values + intercept
                        ax.plot(x_values, trend_line, color=colors[group_idx], 
                            linestyle='--', alpha=0.8, linewidth=2)
                        
                        # Store trend results
                        if trend_analysis:
                            trend_results[f'{group}_{y_col}'] = {
                                'slope': slope,
                                'r_squared': r_value**2,
                                'p_value': p_value,
                                'trend': 'increasing' if slope > 0 else 'decreasing'
                            }
                        
                        # Add trend info to plot
                        ax.text(0.05, 0.95 - group_idx*0.1, 
                            f'{group}: R²={r_value**2:.3f}, p={p_value:.3f}',
                            transform=ax.transAxes, fontsize=9,
                            bbox=dict(boxstyle='round', facecolor=colors[group_idx], alpha=0.3))
                    
                    except Exception as e:
                        print(f"Could not fit trend line for {group} - {y_col}: {e}")

            # Customize subplot - MOVED OUTSIDE THE GROUP LOOP AND FIXED FOR EACH METRIC
            ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{y_col.replace("_", " ").title()} Temporal Trends', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Use subplots_adjust instead of tight_layout
        fig.subplots_adjust(bottom=0.1)
        
        # Store trend results as attribute
        if trend_analysis:
            self.trend_results = trend_results
            
            # Print trend summary
            print("\n" + "="*50)
            print("🔍 TEMPORAL TREND ANALYSIS")
            print("="*50)
            for key, result in trend_results.items():
                group, metric = key.split('_', 1)
                print(f"\n📊 {group} - {metric.replace('_', ' ').title()}:")
                print(f"  Trend: {result['trend']}")
                print(f"  R²: {result['r_squared']:.3f}")
                print(f"  Slope: {result['slope']:.3f}")
                print(f"  P-value: {result['p_value']:.3f}")
        
        # Store figure
        self.temporal_trends_plot = fig
        
        if save_path:
            self.save_publication_plot(self.temporal_trends_plot, save_path)
        
        if show:
            plt.show()
        
        return fig

    def plot_time_series_heatmap(self, x_col: str = 'frame', y_col: str = None,
                           group_col: str = 'group', sample_col: str = 'sample',
                           aggregation: str = 'mean', title: str = None,
                           show: bool = False, save_path: str = None, **kwargs) -> Figure:
        """
        Create a heatmap showing metric values over time for different samples/groups.
        
        Args:
            x_col: Column name for x-axis (time variable)
            y_col: Column name for the metric to visualize
            group_col: Column name for grouping
            sample_col: Column name for sample identification
            aggregation: How to aggregate data ('mean', 'median', 'max', 'min')
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments for seaborn heatmap
            
        Returns:
            matplotlib Figure object
        """
        if y_col is None:
            y_col = self._get_numeric_columns()[0]  # Use first numeric column
        
        # Create pivot table
        if sample_col in self.regionprops.columns:
            # Create sample-time matrix
            pivot_data = self.regionprops.pivot_table(
                values=y_col, 
                index=[group_col, sample_col], 
                columns=x_col, 
                aggfunc=aggregation,
                fill_value=0
            )
        else:
            # Create group-time matrix
            pivot_data = self.regionprops.pivot_table(
                values=y_col,
                index=group_col,
                columns=x_col,
                aggfunc=aggregation,
                fill_value=0
            )
        
        # Create heatmap with figure that handles colorbar properly
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set default heatmap parameters
        heatmap_kwargs = {
            'cmap': 'viridis',
            'annot': False,
            'fmt': '.2f',
            'cbar_kws': {'label': y_col.replace('_', ' ').title()}
        }
        heatmap_kwargs.update(kwargs)
        
        # Create heatmap
        sns.heatmap(pivot_data, ax=ax, **heatmap_kwargs)
        
        # Customize plot
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Sample' if sample_col in self.regionprops.columns else 'Group', fontsize=12)
        
        if title is None:
            title = f'{y_col.replace("_", " ").title()} Heatmap Over Time'
        ax.set_title(title, fontsize=16)
        
        # Use bbox_inches='tight' instead of tight_layout for better colorbar handling
        fig.subplots_adjust(bottom=0.15, left=0.15, right=0.85)
        
        # Store figure
        self.time_series_heatmap = fig
        
        if save_path:
            self.save_publication_plot(self.time_series_heatmap, save_path)
        
        if show:
            plt.show()
        
        return fig

    def plot_multi_metric_time_series(self, x_col: str = 'frame', y_cols: List[str] = None,
                                 group_col: str = 'group', normalize: bool = False,
                                 title: str = None, show: bool = False,
                                 save_path: str = None, **kwargs) -> Figure:
        """
        Create a multi-metric time series plot with all metrics on the same axes.
        
        Args:
            x_col: Column name for x-axis (time variable)
            y_cols: List of column names for metrics to plot
            group_col: Column name for grouping
            normalize: Whether to normalize metrics to 0-1 scale
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the plot
            **kwargs: Additional arguments
            
        Returns:
            matplotlib Figure object
        """
        if y_cols is None:
            y_cols = self._get_numeric_columns()[:5]  # Limit to 5 metrics for readability
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        groups = self.regionprops[group_col].unique()
        n_groups = len(groups)
        n_metrics = len(y_cols)
        
        # Create color map: different hues for groups, different saturations for metrics
        import matplotlib.colors as mcolors
        base_colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
        
        for group_idx, group in enumerate(groups):
            group_data = self.regionprops[self.regionprops[group_col] == group]
            
            # Get metric colors (different saturations of the group color)
            metric_colors = [mcolors.to_rgba(base_colors[group_idx], alpha=0.4 + 0.6*i/n_metrics) 
                            for i in range(n_metrics)]
            
            for metric_idx, y_col in enumerate(y_cols):
                # Aggregate data
                agg_data = group_data.groupby(x_col)[y_col].mean().reset_index()
                
                y_values = agg_data[y_col]
                
                # Normalize if requested
                if normalize:
                    y_min, y_max = y_values.min(), y_values.max()
                    if y_max > y_min:
                        y_values = (y_values - y_min) / (y_max - y_min)
                
                # Plot line
                line_style = '-' if metric_idx < 3 else '--'  # Vary line styles
                ax.plot(agg_data[x_col], y_values, 
                       color=metric_colors[metric_idx], 
                       label=f'{group} - {y_col.replace("_", " ").title()}',
                       linestyle=line_style, linewidth=2, marker='o', markersize=3)
    
        # Customize plot
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Normalized Values' if normalize else 'Metric Values', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Handle legend (might be crowded)
        if len(groups) * len(y_cols) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Set title
        if title is None:
            title = 'Multi-Metric Time Series' + (' (Normalized)' if normalize else '')
        ax.set_title(title, fontsize=16)
        
        # Use subplots_adjust instead of tight_layout
        fig.subplots_adjust(right=0.8)
        
        # Store figure
        self.multi_metric_time_series_plot = fig
        
        if save_path:
            self.save_publication_plot(self.multi_metric_time_series_plot, save_path)
        
        if show:
            plt.show()
        
        return fig
def test_plotting():
    """Test function for the plotting class"""
    # Generate test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'group': ['Control'] * 50 + ['Treatment'] * 50,
        'area': np.concatenate([np.random.normal(100, 20, 50) , np.random.normal(10,20,50)]),
        'perimeter': np.random.normal(50, 10, 100),
        'eccentricity': np.random.uniform(0, 1, 100),
        'solidity': np.concatenate([np.random.uniform(0.8, 1.0, 50),np.random.normal(0.3,0.6,50)]),
        'orientation': np.random.uniform(0, np.pi, 100),  # Add orientation data
        'frame': np.tile(np.arange(1, 11), 10),
        'sample': np.repeat([f'Sample_{i}' for i in range(1, 11)], 10)
    })
    
    # Create plotting instance
    plots = ExperimentPlots(regionprops=test_data)
    
    print("Testing directionality plot...")
    plots.plot_directionality(show=True)
    
    print("Testing directionality rose plot...")
    plots.plot_directionality_rose(show=True)
    
    print("Testing directionality comparison...")
    plots.plot_directionality_comparison(plot_type='violin', show=True)

    # Test different plot types
    print("Testing temporal heatmaps plot...")
    plots.plot_time_series_heatmap(y_col='area')
    print("Testing temporal trends plot...")
    plots.plot_temporal_trends(y_cols=['area', 'perimeter'], show=True)
    print("Testing time series plot...")
    plots.plot_time_series(y_cols=['area', 'perimeter'], show=True)

    print("Testing multi metric timeseries plot...")
    plots.plot_multi_metric_time_series(y_cols=['area', 'perimeter'], show=True, normalize=True)

    print("Testing joint plot...")
    plots.plot_jointplot(x='area', y='perimeter', show=True)
    
    print("Testing pair plot...")
    plots.plot_pairplot(show=True)
    
    print("Testing categorical plot...")
    plots.plot_categorical_comparisons(metric='area', show=True)
    
    print("Testing PCA plot...")
    plots.plot_pca(show=True)
    
    

   
    print("Testing publication save...")
    # Create Test_outputs directory
    os.makedirs("Test_outputs", exist_ok=True)
    
    # Test saving with multiple formats
    plots.save_publication_plot(
        plots.temporal_trends_plot, 
        "Test_outputs/test_publication_plot",
        file_formats=['png', 'pdf', 'svg'],
        dpi=300,
        transparent=True
    )
    
    plots.clear_plots()
    print("All tests completed!")


if __name__ == "__main__":
    test_plotting()