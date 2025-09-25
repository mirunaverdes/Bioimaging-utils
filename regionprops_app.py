import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk
from tkinter import StringVar, BooleanVar, messagebox
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilenames
import os
import numpy as np
import pandas as pd
import gc  # For garbage collection

from experiment import Experiment, ExperimentSample, RegionpropsFilter
from experiment_plots import ExperimentPlots

class ScatterControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master, corner_radius=15)
        self.experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback
        self.plots: ExperimentPlots = None  # Will hold ExperimentPlots instance

        # Configure grid weights
        self.grid_columnconfigure((1, 3, 5), weight=1)

        # Variables
        self.xaxis_var = StringVar()
        self.yaxis_var = StringVar()
        self.title = StringVar()
        self.xlabel = StringVar()
        self.ylabel = StringVar()
        self.plot_kind_var = StringVar(value="scatter")

        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📊 Scatter Plot Configuration", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=6, pady=(10, 15))

        # Main controls row
        ctk.CTkLabel(self, text="X axis:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.xaxis_dropdown = ctk.CTkComboBox(
            self, variable=self.xaxis_var, values=[], width=150,
            font=ctk.CTkFont(size=11)
        )
        self.xaxis_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Y axis:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=2, padx=10, pady=5, sticky="e"
        )
        self.yaxis_dropdown = ctk.CTkComboBox(
            self, variable=self.yaxis_var, values=[], width=150,
            font=ctk.CTkFont(size=11)
        )
        self.yaxis_dropdown.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # Plot kind dropdown
        ctk.CTkLabel(self, text="Kind:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=4, padx=10, pady=5, sticky="e"
        )
        self.plot_kind_dropdown = ctk.CTkComboBox(
            self, variable=self.plot_kind_var, 
            values=["scatter", "kde", "hex", "reg"], 
            width=100, font=ctk.CTkFont(size=11)
        )
        self.plot_kind_dropdown.grid(row=1, column=5, padx=5, pady=5, sticky="ew")

        # Buttons row
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, columnspan=6, padx=10, pady=5, sticky="ew")
        
        self.update_button = ctk.CTkButton(
            button_frame, text="📈 Update Plot", command=self.update_callback,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.update_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(
            button_frame, text="💾 Save Plot", command=self.save_scatterplot,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.save_button.pack(side="left", padx=5)

        # Label customization row
        ctk.CTkLabel(self, text="Title:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=0, padx=10, pady=5, sticky="e"
        )
        self.title_entry = ctk.CTkEntry(
            self, textvariable=self.title, width=150, 
            placeholder_text="Auto-generated from axes"
        )
        self.title_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="X label:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=2, padx=10, pady=5, sticky="e"
        )
        self.xlabel_entry = ctk.CTkEntry(
            self, textvariable=self.xlabel, width=150,
            placeholder_text="Auto-generated from X axis"
        )
        self.xlabel_entry.grid(row=3, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Y label:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=4, padx=10, pady=5, sticky="e"
        )
        self.ylabel_entry = ctk.CTkEntry(
            self, textvariable=self.ylabel, width=150,
            placeholder_text="Auto-generated from Y axis"
        )
        self.ylabel_entry.grid(row=3, column=5, padx=5, pady=5, sticky="ew")

    def update_dropdowns(self, numeric_cols):
        self.xaxis_dropdown.configure(values=numeric_cols)
        self.yaxis_dropdown.configure(values=numeric_cols)
        if numeric_cols:
            if not self.xaxis_var.get():
                self.xaxis_var.set(numeric_cols[0])
            if not self.yaxis_var.get() and len(numeric_cols) > 1:
                self.yaxis_var.set(numeric_cols[1])

    def save_scatterplot(self):
        """Save scatter plot with error handling"""
        try:
            filepath = asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filepath and self.plots and hasattr(self.plots, 'joint_plot') and self.plots.joint_plot:
                self.plots.save_publication_plot(self.plots.joint_plot, filepath=filepath)
                print(f"Plot saved to: {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

class CategoricalControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master, corner_radius=15)
        self.experiment: Experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback
        self.plots : ExperimentPlots = None  # Will hold ExperimentPlots instance

        # Configure grid weights
        self.grid_columnconfigure((1, 3), weight=1)

        # Variables
        self.plotType_var = StringVar(value="box")
        self.metric_var = StringVar()
        self.title = StringVar()
        self.xlabel = StringVar()
        self.ylabel = StringVar()
        self.annotate_var = BooleanVar(value=True)

        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📊 Categorical Comparison Configuration", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=6, pady=(10, 15))

        # Main controls row
        ctk.CTkLabel(self, text="Plot type:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.plotType_dropdown = ctk.CTkComboBox(
            self, variable=self.plotType_var, 
            values=["box", "violin", "boxen", "strip", "point", "count", "bar"], 
            width=150, font=ctk.CTkFont(size=11)
        )
        self.plotType_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Metric:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=2, padx=10, pady=5, sticky="e"
        )
        self.metric_dropdown = ctk.CTkComboBox(
            self, variable=self.metric_var, values=[], width=200,
            font=ctk.CTkFont(size=11)
        )
        self.metric_dropdown.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # Statistical annotations checkbox
        self.annotate_check = ctk.CTkCheckBox(
            self, text="📈 Statistical annotations", variable=self.annotate_var,
            font=ctk.CTkFont(size=11)
        )
        self.annotate_check.grid(row=1, column=4, columnspan=2, padx=10, pady=5, sticky="w")

        # Buttons row
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, columnspan=6, padx=10, pady=5, sticky="ew")
        
        self.update_button = ctk.CTkButton(
            button_frame, text="📈 Update Plot", command=self.update_callback,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.update_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(
            button_frame, text="💾 Save Plot", command=self.save_cat_plot,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.save_button.pack(side="left", padx=5)

        # Label customization row
        ctk.CTkLabel(self, text="Title:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=0, padx=10, pady=5, sticky="e"
        )
        self.title_entry = ctk.CTkEntry(
            self, textvariable=self.title, width=150,
            placeholder_text="Auto-generated from metric"
        )
        self.title_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="X label:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=2, padx=10, pady=5, sticky="e"
        )
        self.xlabel_entry = ctk.CTkEntry(
            self, textvariable=self.xlabel, width=150,
            placeholder_text="Group"
        )
        self.xlabel_entry.grid(row=3, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Y label:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=4, padx=10, pady=5, sticky="e"
        )
        self.ylabel_entry = ctk.CTkEntry(
            self, textvariable=self.ylabel, width=150,
            placeholder_text="Auto-generated from metric"
        )
        self.ylabel_entry.grid(row=3, column=5, padx=5, pady=5, sticky="ew")

    def update_dropdowns(self, numeric_cols):
        self.metric_dropdown.configure(values=numeric_cols)
        if numeric_cols and not self.metric_var.get():
            self.metric_var.set(numeric_cols[0])

    def save_cat_plot(self):
        """Save categorical plot with error handling"""
        try:
            filepath = asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filepath and self.plots and hasattr(self.plots, 'categorical_plot') and self.plots.categorical_plot:
                self.plots.save_publication_plot(self.plots.categorical_plot, filepath=filepath)
                print(f"Plot saved to: {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

class PairplotControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master, corner_radius=15)
        self.experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback
        self.plots : ExperimentPlots = None  # Will hold ExperimentPlots instance

        # Configure grid weights
        self.grid_columnconfigure((1, 3), weight=1)

        # Variables
        self.selected_columns = []
        self.title = StringVar()
        self.corner_var = BooleanVar(value=True)
        self.height_var = StringVar(value="2.5")

        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📊 Pairplot Configuration", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=6, pady=(10, 15))

        # Column selection info
        self.columns_info_label = ctk.CTkLabel(
            self, 
            text="📋 Select columns for pairplot (up to 6 recommended)", 
            font=ctk.CTkFont(size=12)
        )
        self.columns_info_label.grid(row=1, column=0, columnspan=6, pady=5)

        # Scrollable frame for column checkboxes
        self.columns_frame = ctk.CTkScrollableFrame(self, height=40)
        self.columns_frame._scrollbar.configure(height=5)  # Shorter scrollbar
        self.columns_frame.grid(row=2, column=0, columnspan=6, sticky="ew", padx=10, pady=5)

        # Options row
        ctk.CTkLabel(self, text="Corner:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=3, column=0, padx=10, pady=5, sticky="e"
        )
        self.corner_check = ctk.CTkCheckBox(
            self, text="Show only lower triangle", variable=self.corner_var,
            font=ctk.CTkFont(size=11)
        )
        self.corner_check.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Height:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=3, column=2, padx=10, pady=5, sticky="e"
        )
        self.height_entry = ctk.CTkEntry(
            self, textvariable=self.height_var, width=80,
            placeholder_text="2.5"
        )
        self.height_entry.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # Buttons row
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=4, column=0, columnspan=6, padx=10, pady=5, sticky="ew")
        
        self.update_button = ctk.CTkButton(
            button_frame, text="📈 Update Plot", command=self.update_callback,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.update_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(
            button_frame, text="💾 Save Plot", command=self.save_pairplot,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.save_button.pack(side="left", padx=5)

        # Title customization
        ctk.CTkLabel(self, text="Title:", font=ctk.CTkFont(size=11)).grid(
            row=5, column=0, padx=10, pady=5, sticky="e"
        )
        self.title_entry = ctk.CTkEntry(
            self, textvariable=self.title, width=300,
            placeholder_text="Auto-generated"
        )
        self.title_entry.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

    def update_columns(self, numeric_cols):
        """Update available columns for selection"""
        # Clear existing checkboxes
        for widget in self.columns_frame.winfo_children():
            widget.destroy()
        
        self.column_vars = []
        
        # Create checkboxes for each numeric column
        for i, col in enumerate(numeric_cols):
            var = BooleanVar(value=(i < 6))  # Select first 6 by default
            self.column_vars.append((col, var))
            
            checkbox = ctk.CTkCheckBox(
                self.columns_frame, 
                text=col.replace('_', ' ').title(), 
                variable=var,
                font=ctk.CTkFont(size=11)
            )
            checkbox.grid(row=i//5, column=i%5, sticky="w", padx=5, pady=2)

    def get_selected_columns(self):
        """Get list of selected columns"""
        return [col for col, var in self.column_vars if var.get()]

    def save_pairplot(self):
        """Save pairplot with error handling"""
        try:
            filepath = asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filepath and self.plots and hasattr(self.plots, 'pair_plot') and self.plots.pair_plot:
                self.plots.save_publication_plot(self.plots.pair_plot, filepath=filepath)
                print(f"Plot saved to: {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

class PCAControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master, corner_radius=15)
        self.experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback
        self.plots : ExperimentPlots = None  # Will hold ExperimentPlots instance

        # Variables
        self.n_components_var = StringVar(value="2")
        self.title = StringVar()

        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📊 PCA Configuration", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(10, 15))

        # Components selection
        ctk.CTkLabel(self, text="Components:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.components_dropdown = ctk.CTkComboBox(
            self, variable=self.n_components_var, 
            values=["2", "3"], 
            width=100, font=ctk.CTkFont(size=11)
        )
        self.components_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Info label
        info_label = ctk.CTkLabel(
            self, 
            text="Uses all numeric columns (automatically scaled)", 
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="gray"
        )
        info_label.grid(row=1, column=2, columnspan=2, padx=10, pady=5, sticky="w")

        # Buttons row
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5, sticky="ew")
        
        self.update_button = ctk.CTkButton(
            button_frame, text="📈 Update Plot", command=self.update_callback,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.update_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(
            button_frame, text="💾 Save Plot", command=self.save_pca_plot,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.save_button.pack(side="left", padx=5)

        # Title customization
        ctk.CTkLabel(self, text="Title:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=0, padx=10, pady=5, sticky="e"
        )
        self.title_entry = ctk.CTkEntry(
            self, textvariable=self.title, width=300,
            placeholder_text="Auto-generated with variance explained"
        )
        self.title_entry.grid(row=3, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Configure grid weights
        self.grid_columnconfigure(1, weight=1)

    def save_pca_plot(self):
        """Save PCA plot with error handling"""
        try:
            filepath = asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filepath and self.plots and hasattr(self.plots, 'pca_plot') and self.plots.pca_plot:
                self.plots.save_publication_plot(self.plots.pca_plot, filepath=filepath)
                print(f"Plot saved to: {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

class TimeSeriesControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master, corner_radius=15)
        self.experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback
        self.plots : ExperimentPlots = None  # Will hold ExperimentPlots instance

        # Configure grid weights
        self.grid_columnconfigure((1, 3, 5), weight=1)

        # Variables
        self.x_col_var = StringVar(value="frame")
        self.y_cols_selected = []
        self.aggregation_var = StringVar(value="mean")
        self.error_bars_var = StringVar(value="std")
        self.title = StringVar()
        self.xlabel = StringVar()
        self.ylabel = StringVar()
        self.show_individual_var = BooleanVar(value=False)
        self.smooth_var = BooleanVar(value=False)
        self.smooth_window_var = StringVar(value="3")

        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📈 Time Series Plot Configuration", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=6, pady=(10, 15))

        # Main controls row 1
        ctk.CTkLabel(self, text="Time axis:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.x_col_dropdown = ctk.CTkComboBox(
            self, variable=self.x_col_var, 
            values=["frame", "time"], width=120,
            font=ctk.CTkFont(size=11)
        )
        self.x_col_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Aggregation:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=2, padx=10, pady=5, sticky="e"
        )
        self.aggregation_dropdown = ctk.CTkComboBox(
            self, variable=self.aggregation_var, 
            values=["mean", "median", "sum"], width=100,
            font=ctk.CTkFont(size=11)
        )
        self.aggregation_dropdown.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Error bars:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=4, padx=10, pady=5, sticky="e"
        )
        self.error_bars_dropdown = ctk.CTkComboBox(
            self, variable=self.error_bars_var, 
            values=["std", "sem", "ci", "none"], width=100,
            font=ctk.CTkFont(size=11)
        )
        self.error_bars_dropdown.grid(row=1, column=5, padx=5, pady=5, sticky="ew")

        # Y-axis metrics selection
        ctk.CTkLabel(self, text="Metrics:", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=2, column=0, padx=10, pady=5, sticky="ne"
        )
        
        # Scrollable frame for metric checkboxes
        self.metrics_frame = ctk.CTkScrollableFrame(self, height=40)
        self.metrics_frame._scrollbar.configure(height=5)  # Shorter scrollbar
        self.metrics_frame.grid(row=2, column=1, columnspan=6, sticky="ew", padx=5, pady=5)

        # Options row
        options_frame = ctk.CTkFrame(self, fg_color="transparent")
        options_frame.grid(row=3, column=0, columnspan=6, sticky="ew", padx=10, pady=5)
        
        self.show_individual_check = ctk.CTkCheckBox(
            options_frame, text="📊 Show individual traces", 
            variable=self.show_individual_var,
            font=ctk.CTkFont(size=11)
        )
        self.show_individual_check.pack(side="left", padx=5)
        
        self.smooth_check = ctk.CTkCheckBox(
            options_frame, text="🌊 Smooth data", 
            variable=self.smooth_var,
            font=ctk.CTkFont(size=11)
        )
        self.smooth_check.pack(side="left", padx=5)
        
        ctk.CTkLabel(options_frame, text="Window:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(10, 2))
        self.smooth_window_entry = ctk.CTkEntry(
            options_frame, textvariable=self.smooth_window_var, width=50
        )
        self.smooth_window_entry.pack(side="left", padx=2)

        # Buttons row
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=4, column=0, columnspan=6, padx=10, pady=5, sticky="ew")
        
        self.update_button = ctk.CTkButton(
            button_frame, text="📈 Update Plot", command=self.update_callback,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.update_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(
            button_frame, text="💾 Save Plot", command=self.save_time_series_plot,
            font=ctk.CTkFont(size=12, weight="bold"), height=32
        )
        self.save_button.pack(side="left", padx=5)

        # Advanced plot types
        # advanced_frame = ctk.CTkFrame(self, fg_color="transparent")
        # advanced_frame.grid(row=5, column=0, columnspan=6, sticky="ew", padx=10, pady=5)
        
        # self.heatmap_button = ctk.CTkButton(
        #     advanced_frame, text="🔥 Time Heatmap", command=self.create_heatmap,
        #     font=ctk.CTkFont(size=11, weight="bold"), height=28,
        #     fg_color="orange", hover_color="darkorange"
        # )
        # self.heatmap_button.pack(side="left", padx=5)
        
        # self.trends_button = ctk.CTkButton(
        #     advanced_frame, text="📊 Trend Analysis", command=self.create_trends,
        #     font=ctk.CTkFont(size=11, weight="bold"), height=28,
        #     fg_color="purple", hover_color="darkpurple"
        # )
        # self.trends_button.pack(side="left", padx=5)
        
        # self.multi_metric_button = ctk.CTkButton(
        #     advanced_frame, text="📈 Multi-Metric", command=self.create_multi_metric,
        #     font=ctk.CTkFont(size=11, weight="bold"), height=28,
        #     fg_color="teal", hover_color="darkteal"
        # )
        # self.multi_metric_button.pack(side="left", padx=5)

        # Label customization row
        ctk.CTkLabel(self, text="Title:", font=ctk.CTkFont(size=11)).grid(
            row=6, column=0, padx=10, pady=5, sticky="e"
        )
        self.title_entry = ctk.CTkEntry(
            self, textvariable=self.title, width=150,
            placeholder_text="Auto-generated"
        )
        self.title_entry.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="X label:", font=ctk.CTkFont(size=11)).grid(
            row=6, column=2, padx=10, pady=5, sticky="e"
        )
        self.xlabel_entry = ctk.CTkEntry(
            self, textvariable=self.xlabel, width=150,
            placeholder_text="Auto-generated from time axis"
        )
        self.xlabel_entry.grid(row=6, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Y label:", font=ctk.CTkFont(size=11)).grid(
            row=6, column=4, padx=10, pady=5, sticky="e"
        )
        self.ylabel_entry = ctk.CTkEntry(
            self, textvariable=self.ylabel, width=150,
            placeholder_text="Auto-generated from metrics"
        )
        self.ylabel_entry.grid(row=6, column=5, padx=5, pady=5, sticky="ew")

        # Initialize
        self.metric_vars = []

    def update_metrics(self, numeric_cols):
        """Update available metrics for selection"""
        # Clear existing checkboxes
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        self.metric_vars = []
        
        # Add time columns to x-axis dropdown if available
        time_cols = ["frame", "time"]
        available_time_cols = [col for col in time_cols if col in self.experiment.regionprops.columns]
        if available_time_cols:
            self.x_col_dropdown.configure(values=available_time_cols)
            if not self.x_col_var.get() in available_time_cols:
                self.x_col_var.set(available_time_cols[0])
        
        # Create checkboxes for each numeric column
        for i, col in enumerate(numeric_cols):
            var = BooleanVar(value=(i < 3))  # Select first 3 by default
            self.metric_vars.append((col, var))
            
            checkbox = ctk.CTkCheckBox(
                self.metrics_frame, 
                text=col.replace('_', ' ').title(), 
                variable=var,
                font=ctk.CTkFont(size=11)
            )
            # Arrange in a grid-like manner
            checkbox.grid(row=i//5, column=i%5, sticky="w", padx=5, pady=2)

    def get_selected_metrics(self):
        """Get list of selected metrics"""
        return [col for col, var in self.metric_vars if var.get()]

    def save_time_series_plot(self):
        """Save time series plot with error handling"""
        try:
            filepath = asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filepath and self.plots and hasattr(self.plots, 'time_series_plot') and self.plots.time_series_plot:
                self.plots.save_publication_plot(self.plots.time_series_plot, filepath=filepath)
                print(f"Plot saved to: {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    def create_heatmap(self):
        """Create time series heatmap"""
        if not self.plots:
            return
        
        selected_metrics = self.get_selected_metrics()
        if not selected_metrics:
            return
        
        x_col = self.x_col_var.get()
        y_col = selected_metrics[0]  # Use first selected metric
        
        try:
            fig = self.plots.plot_time_series_heatmap(
                x_col=x_col, y_col=y_col, show=False
            )
            self.update_callback()  # This will refresh the current plot display
            print(f"✅ Heatmap created for {y_col} over {x_col}")
        except Exception as e:
            print(f"Error creating heatmap: {e}")

    def create_trends(self):
        """Create temporal trends plot"""
        if not self.plots:
            return
        
        selected_metrics = self.get_selected_metrics()
        if not selected_metrics:
            return
        
        x_col = self.x_col_var.get()
        
        try:
            fig = self.plots.plot_temporal_trends(
                x_col=x_col, y_cols=selected_metrics[:2], show=False  # Limit to 2 for readability
            )
            self.update_callback()  # This will refresh the current plot display
            print(f"✅ Trend analysis created for {selected_metrics[:2]}")
        except Exception as e:
            print(f"Error creating trends plot: {e}")

    def create_multi_metric(self):
        """Create multi-metric time series plot"""
        if not self.plots:
            return
        
        selected_metrics = self.get_selected_metrics()
        if not selected_metrics:
            return
        
        x_col = self.x_col_var.get()
        
        try:
            fig = self.plots.plot_multi_metric_time_series(
                x_col=x_col, y_cols=selected_metrics, normalize=True, show=False
            )
            self.update_callback()  # This will refresh the current plot display
            print(f"✅ Multi-metric plot created with {len(selected_metrics)} metrics")
        except Exception as e:
            print(f"Error creating multi-metric plot: {e}")

class StatsTabFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=15)
        
        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📈 Statistical Analysis Results", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(10, 15))

        # Main tests section
        tests_label = ctk.CTkLabel(
            self, 
            text="Overall Tests", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        tests_label.pack(pady=(10, 5))

        self.stat_tree = ttk.Treeview(
            self, columns=("Test", "Statistic", "p-value"), 
            show="headings", height=5
        )
        self.stat_tree.heading("Test", text="Test")
        self.stat_tree.heading("Statistic", text="Statistic")
        self.stat_tree.heading("p-value", text="p-value")
        
        # Style the treeview
        self.stat_tree.column("Test", width=150)
        self.stat_tree.column("Statistic", width=120)
        self.stat_tree.column("p-value", width=120)
        
        self.stat_tree.pack(fill="x", expand=False, padx=10, pady=5)

        # Post-hoc tests section
        posthoc_label = ctk.CTkLabel(
            self, 
            text="Post-hoc Comparisons", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        posthoc_label.pack(pady=(15, 5))

        self.posthoc_tree = ttk.Treeview(
            self, columns=("group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"), 
            show="headings", height=10
        )
        
        headers = {
            "group1": "Group 1",
            "group2": "Group 2", 
            "meandiff": "Mean Diff",
            "p-adj": "p-adj",
            "lower": "Lower CI",
            "upper": "Upper CI",
            "reject": "Significant"
        }
        
        for col, header in headers.items():
            self.posthoc_tree.heading(col, text=header)
            self.posthoc_tree.column(col, width=100)
        
        self.posthoc_tree.pack(fill="both", expand=True, padx=10, pady=5)

    def update_stats(self, experiment, metric):
        # Clear existing data
        for row in self.stat_tree.get_children():
            self.stat_tree.delete(row)
        for row in self.posthoc_tree.get_children():
            self.posthoc_tree.delete(row)
        
        # Update main tests
        if hasattr(experiment, "anova_results") and experiment.anova_results is not None:
            self.stat_tree.insert("", "end", values=(
                "ANOVA", 
                f"{experiment.anova_results.statistic:.4g}", 
                f"{experiment.anova_results.pvalue:.4g}"
            ))
        
        if hasattr(experiment, "kruskal_results") and experiment.kruskal_results is not None:
            self.stat_tree.insert("", "end", values=(
                "Kruskal-Wallis", 
                f"{experiment.kruskal_results.statistic:.4g}", 
                f"{experiment.kruskal_results.pvalue:.4g}"
            ))
        
        # Update post-hoc tests
        if hasattr(experiment, "tukey_results") and hasattr(experiment.tukey_results, "empty") and not experiment.tukey_results.empty:
            for _, row in experiment.tukey_results.iterrows():
                self.posthoc_tree.insert("", "end", values=tuple(row))

class GroupFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, update_callback, remove_callback):
        super().__init__(master, corner_radius=15)  # Remove fixed width
        self.experiment = experiment
        self.update_callback = update_callback
        self.remove_callback = remove_callback
        
        # Configure grid to be responsive
        self.grid_columnconfigure(1, weight=1)
        
        # Variables
        self.group_name_var = StringVar()
        self.regionprops_paths = []
        self.check_vars = []
        self.checkboxes = []
        
        # Group name section
        name_frame = ctk.CTkFrame(self, fg_color="transparent")
        name_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        name_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            name_frame, text="🏷️ Group:", 
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, padx=2, pady=5, sticky="w")
        
        self.group_name_entry = ctk.CTkEntry(
            name_frame, textvariable=self.group_name_var,
            placeholder_text="Enter group name",
            font=ctk.CTkFont(size=11),
            height=28
        )
        self.group_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Browse button
        self.browse_button = ctk.CTkButton(
            name_frame, text="📂 Browse", command=self.browse_regionprops,
            font=ctk.CTkFont(size=11, weight="bold"), height=28,
            width=80
        )
        self.browse_button.grid(row=0, column=2, padx=2, pady=5)
        
        # Files list frame with responsive sizing
        self.files_frame = ctk.CTkScrollableFrame(
            self, height=40, corner_radius=8
        )
        self.files_frame._scrollbar.configure(height=5)  # Shorter scrollbar
        self.files_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.grid(row=2, column=0, columnspan=4, pady=5)
        
        self.add_button = ctk.CTkButton(
            buttons_frame, text="✅ Add", command=self.add_group,
            state="disabled", font=ctk.CTkFont(size=11, weight="bold"),
            height=30, width=80
        )
        self.add_button.pack(side="left", padx=3)
        
        self.remove_button = ctk.CTkButton(
            buttons_frame, text="❌ Remove", command=self.remove_group,
            state="disabled", font=ctk.CTkFont(size=11, weight="bold"),
            height=30, fg_color="red", hover_color="darkred", width=80
        )
        self.remove_button.pack(side="left", padx=3)
        
        # Bind events
        self.group_name_var.trace_add("write", self.check_ready_state)

    def check_ready_state(self, *args):
        """Check if group is ready to be added"""
        has_name = bool(self.group_name_var.get().strip())
        has_files = any(var.get() for var in self.check_vars)
        
        if has_name and has_files:
            self.add_button.configure(state="normal")
            self.remove_button.configure(state="normal")
        else:
            self.add_button.configure(state="disabled")
    
    def browse_regionprops(self):
        """Browse for regionprops files"""
        # Clear existing file widgets
        for widget in self.files_frame.winfo_children():
            widget.destroy()
        self.checkboxes.clear()
        self.check_vars.clear()
        self.regionprops_paths.clear()
        
        paths = askopenfilenames(
            title=f"Select regionprops files for {self.group_name_var.get() or 'this group'}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not paths:
            return
        
        # Add file checkboxes with better sizing
        for i, path in enumerate(paths):
            var = BooleanVar(value=True)
            self.check_vars.append(var)
            self.regionprops_paths.append(path)
            
            filename = os.path.basename(path)
            # Truncate long filenames for better display
            display_name = filename if len(filename) < 45 else f"{filename[:35]}...{filename[-7:]}"
            
            cb = ctk.CTkCheckBox(
                self.files_frame, text=display_name, variable=var,
                command=self.check_ready_state, font=ctk.CTkFont(size=10)
            )
            cb.pack(anchor="w", padx=3, pady=1)
            self.checkboxes.append(cb)
        
        self.check_ready_state()
    
    def add_group(self):
        """Add group to experiment"""
        group_name = self.group_name_var.get().strip()
        selected_paths = [p for p, v in zip(self.regionprops_paths, self.check_vars) if v.get()]
        
        for path in selected_paths:
            sample_name = os.path.basename(path).replace('_regionprops.csv', '').replace('.csv', '')
            sample = ExperimentSample(
                name=sample_name, 
                group=group_name, 
                regionprops_df_path=path, 
                bitDepth=16, 
                normalize=True
            )
            self.experiment.add_sample(sample)
        
        # Disable editing
        self.group_name_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.add_button.configure(state="disabled")
        for cb in self.checkboxes:
            cb.configure(state="disabled")
        
        # Update dropdowns
        self.update_callback()
    
    def remove_group(self):
        """Remove this group frame"""
        group_name = self.group_name_var.get().strip()
        self.experiment.remove_group(group_name)
        self.remove_callback(self)

class FilterControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, update_callback):
        super().__init__(master, corner_radius=15)
        self.experiment = experiment
        self.update_callback = update_callback
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="🔍 Data Filters", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 15))
        
        # Info label
        self.info_label = ctk.CTkLabel(
            self, 
            text="No data loaded", 
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.info_label.grid(row=1, column=0, pady=5)
        
        # Filter creation frame
        self.create_filter_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.create_filter_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.create_filter_frame.grid_columnconfigure((1, 3, 5), weight=1)
        
        # Filter type selection
        ctk.CTkLabel(
            self.create_filter_frame, text="Type:", 
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        
        self.filter_type_var = StringVar(value="numeric")
        self.filter_type_dropdown = ctk.CTkComboBox(
            self.create_filter_frame, variable=self.filter_type_var,
            values=["numeric", "categorical"], width=120,
            command=self.on_filter_type_change
        )
        self.filter_type_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Column selection
        ctk.CTkLabel(
            self.create_filter_frame, text="Column:", 
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=2, padx=5, pady=5, sticky="e")
        
        self.column_var = StringVar()
        self.column_dropdown = ctk.CTkComboBox(
            self.create_filter_frame, variable=self.column_var,
            values=[], width=150, command=self.on_column_change
        )
        self.column_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Add filter button
        self.add_filter_button = ctk.CTkButton(
            self.create_filter_frame, text="➕ Add Filter",
            command=self.add_filter, width=100,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.add_filter_button.grid(row=0, column=4, padx=5, pady=5)
        
        # Numeric filter controls
        self.numeric_frame = ctk.CTkFrame(self.create_filter_frame)
        self.numeric_frame.grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        self.numeric_frame.grid_columnconfigure((1, 3), weight=1)
        
        ctk.CTkLabel(
            self.numeric_frame, text="Min:", 
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        
        self.min_var = StringVar()
        self.min_entry = ctk.CTkEntry(
            self.numeric_frame, textvariable=self.min_var,
            placeholder_text="Leave empty for no minimum", width=150
        )
        self.min_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            self.numeric_frame, text="Max:", 
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=2, padx=5, pady=5, sticky="e")
        
        self.max_var = StringVar()
        self.max_entry = ctk.CTkEntry(
            self.numeric_frame, textvariable=self.max_var,
            placeholder_text="Leave empty for no maximum", width=150
        )
        self.max_entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Statistical info for numeric columns
        self.stats_label = ctk.CTkLabel(
            self.numeric_frame, text="", 
            font=ctk.CTkFont(size=10), text_color="gray"
        )
        self.stats_label.grid(row=1, column=0, columnspan=4, pady=5)
        
        # Categorical filter controls
        self.categorical_frame = ctk.CTkFrame(self.create_filter_frame)
        self.categorical_frame.grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        self.categorical_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            self.categorical_frame, text="Action:", 
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        
        self.cat_action_var = StringVar(value="keep")
        self.cat_action_dropdown = ctk.CTkComboBox(
            self.categorical_frame, variable=self.cat_action_var,
            values=["keep", "remove"], width=100
        )
        self.cat_action_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Category selection (scrollable frame for checkboxes)
        self.categories_frame = ctk.CTkScrollableFrame(self.categorical_frame, height=80)
        self.categories_frame._scrollbar.configure(height=5)  # Shorter scrollbar
        self.categories_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Hide categorical frame initially
        self.categorical_frame.grid_remove()
        
        # Active filters section
        filters_label = ctk.CTkLabel(
            self, 
            text="📋 Active Filters", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        filters_label.grid(row=3, column=0, pady=(20, 10))
        
        # Filters list frame
        self.filters_list_frame = ctk.CTkScrollableFrame(self, height=150)
        self.filters_list_frame._scrollbar.configure(height=5)  # Shorter scrollbar
        self.filters_list_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        
        # Control buttons
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.grid(row=5, column=0, pady=10)
        
        self.preview_button = ctk.CTkButton(
            buttons_frame, text="👁️ Preview Filters",
            command=self.preview_filters,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.preview_button.pack(side="left", padx=5)
        
        self.apply_button = ctk.CTkButton(
            buttons_frame, text="✅ Apply Filters",
            command=self.apply_filters,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.apply_button.pack(side="left", padx=5)
        
        self.reset_button = ctk.CTkButton(
            buttons_frame, text="🔄 Reset Data",
            command=self.reset_filters,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="orange", hover_color="darkorange"
        )
        self.reset_button.pack(side="left", padx=5)
        
        # Initialize
        self.category_vars = []
        self.update_available_columns()
    
    def show_message(self, title, message, icon="info"):
        """Show message using available method"""
        try:
            # Try CTkMessagebox first (newer versions)
            if hasattr(ctk, 'CTkMessagebox'):
                return ctk.CTkMessagebox(title=title, message=message, icon=icon).get()
            else:
                # Fall back to tkinter messagebox
                if icon == "warning":
                    return messagebox.showwarning(title, message)
                elif icon == "cancel" or icon == "error":
                    return messagebox.showerror(title, message)
                elif icon == "check":
                    return messagebox.showinfo(title, message)
                else:
                    return messagebox.showinfo(title, message)
        except:
            # Ultimate fallback - print to console
            print(f"{title}: {message}")
    
    def update_available_columns(self):
        """Update available columns based on current data"""
        if not self.experiment.regionprops.empty:
            # Get all columns
            all_cols = self.experiment.regionprops.columns.tolist()
            # Remove system columns
            exclude_cols = ["sample", "group"]
            available_cols = [col for col in all_cols if col not in exclude_cols]
            
            self.column_dropdown.configure(values=available_cols)
            
            # Update info
            n_rows = len(self.experiment.regionprops)
            n_groups = self.experiment.regionprops['group'].nunique()
            n_samples = self.experiment.regionprops['sample'].nunique()
            
            self.info_label.configure(
                text=f"📊 {n_rows:,} cells across {n_groups} groups ({n_samples} samples)"
            )
        else:
            self.column_dropdown.configure(values=[])
            self.info_label.configure(text="No data loaded")
    
    def on_filter_type_change(self, value=None):
        """Handle filter type change"""
        if self.filter_type_var.get() == "numeric":
            self.numeric_frame.grid()
            self.categorical_frame.grid_remove()
        else:
            self.numeric_frame.grid_remove()
            self.categorical_frame.grid()
            self.update_categories()
    
    def on_column_change(self, value=None):
        """Handle column selection change"""
        column = self.column_var.get()
        if not column or self.experiment.regionprops.empty:
            return
        
        # Update stats for numeric columns
        if self.filter_type_var.get() == "numeric":
            try:
                col_data = self.experiment.regionprops[column]
                if pd.api.types.is_numeric_dtype(col_data):
                    stats_text = (f"Range: {col_data.min():.3g} - {col_data.max():.3g}, "
                                f"Mean: {col_data.mean():.3g}, "
                                f"Std: {col_data.std():.3g}")
                    self.stats_label.configure(text=stats_text)
                else:
                    self.stats_label.configure(text="Warning: Selected column is not numeric")
            except Exception as e:
                self.stats_label.configure(text=f"Error: {e}")
        
        # Update categories for categorical columns
        elif self.filter_type_var.get() == "categorical":
            self.update_categories()
    
    def update_categories(self):
        """Update category checkboxes"""
        column = self.column_var.get()
        if not column or self.experiment.regionprops.empty:
            return
        
        # Clear existing checkboxes
        for widget in self.categories_frame.winfo_children():
            widget.destroy()
        self.category_vars = []
        
        # Get unique values
        try:
            unique_values = self.experiment.regionprops[column].unique()
            unique_values = sorted([str(v) for v in unique_values if pd.notna(v)])
            
            for value in unique_values:
                var = BooleanVar(value=True)
                self.category_vars.append((value, var))
                
                checkbox = ctk.CTkCheckBox(
                    self.categories_frame, 
                    text=value,
                    variable=var,
                    font=ctk.CTkFont(size=11)
                )
                checkbox.pack(anchor="w", padx=5, pady=2)
                
        except Exception as e:
            error_label = ctk.CTkLabel(
                self.categories_frame,
                text=f"Error loading categories: {e}",
                text_color="red"
            )
            error_label.pack(pady=5)
    
    def add_filter(self):
        """Add a new filter"""
        column = self.column_var.get()
        filter_type = self.filter_type_var.get()
        
        if not column:
            self.show_message("Error", "Please select a column to filter.", "warning")
            return
        
        try:
            if filter_type == "numeric":
                # Get numeric thresholds
                min_val = None
                max_val = None
                
                if self.min_var.get().strip():
                    min_val = float(self.min_var.get())
                if self.max_var.get().strip():
                    max_val = float(self.max_var.get())
                
                if min_val is None and max_val is None:
                    self.show_message("Error", "Please specify at least one threshold.", "warning")
                    return
                
                filter_obj = RegionpropsFilter(
                    column=column, filter_type="numeric",
                    threshold_low=min_val, threshold_high=max_val
                )
                
            else:  # categorical
                # Get selected categories
                action = self.cat_action_var.get()
                selected_categories = [cat for cat, var in self.category_vars if var.get()]
                
                if not selected_categories:
                    self.show_message("Error", "Please select at least one category.", "warning")
                    return
                
                if action == "keep":
                    filter_obj = RegionpropsFilter(
                        column=column, filter_type="categorical",
                        categories_keep=selected_categories
                    )
                else:  # remove
                    filter_obj = RegionpropsFilter(
                        column=column, filter_type="categorical",
                        categories_remove=selected_categories
                    )
            
            # Add filter to experiment
            self.experiment.add_filter(filter_obj)
            self.update_filters_display()
            
            # Clear form
            self.min_var.set("")
            self.max_var.set("")
            
            # Reset category checkboxes
            for _, var in self.category_vars:
                var.set(True)
                
        except ValueError as e:
            self.show_message("Error", f"Invalid input: {e}", "error")
        except Exception as e:
            self.show_message("Error", f"Error adding filter: {e}", "error")
    
    def update_filters_display(self):
        """Update the display of active filters"""
        # Clear existing widgets
        for widget in self.filters_list_frame.winfo_children():
            widget.destroy()
        
        if not hasattr(self.experiment, 'active_filters') or not self.experiment.active_filters:
            no_filters_label = ctk.CTkLabel(
                self.filters_list_frame,
                text="No active filters",
                text_color="gray"
            )
            no_filters_label.pack(pady=10)
            return
        
        # Display each filter
        for i, filter_obj in enumerate(self.experiment.active_filters):
            filter_frame = ctk.CTkFrame(self.filters_list_frame)
            filter_frame.pack(fill="x", padx=5, pady=2)
            filter_frame.grid_columnconfigure(1, weight=1)
            
            # Filter description
            if filter_obj.filter_type == "numeric":
                desc_parts = [f"Column: {filter_obj.column}"]
                if filter_obj.threshold_low is not None:
                    desc_parts.append(f"≥ {filter_obj.threshold_low}")
                if filter_obj.threshold_high is not None:
                    desc_parts.append(f"≤ {filter_obj.threshold_high}")
                description = " | ".join(desc_parts)
            else:
                action = "Keep" if filter_obj.categories_keep else "Remove"
                categories = filter_obj.categories_keep or filter_obj.categories_remove
                description = f"{action} {filter_obj.column}: {', '.join(categories[:3])}"
                if len(categories) > 3:
                    description += f"... (+{len(categories)-3} more)"
            
            desc_label = ctk.CTkLabel(
                filter_frame, text=description,
                font=ctk.CTkFont(size=11)
            )
            desc_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            
            # Enable/disable checkbox
            enabled_var = BooleanVar(value=filter_obj.enabled)
            enabled_check = ctk.CTkCheckBox(
                filter_frame, text="Enabled", variable=enabled_var,
                command=lambda i=i, var=enabled_var: self.toggle_filter(i, var)
            )
            enabled_check.grid(row=0, column=1, padx=5, pady=5)
            
            # Remove button
            remove_button = ctk.CTkButton(
                filter_frame, text="❌", width=30,
                command=lambda i=i: self.remove_filter(i),
                fg_color="red", hover_color="darkred"
            )
            remove_button.grid(row=0, column=2, padx=5, pady=5)
    
    def toggle_filter(self, index, var):
        """Toggle filter enabled state"""
        if hasattr(self.experiment, 'active_filters') and index < len(self.experiment.active_filters):
            self.experiment.active_filters[index].enabled = var.get()
    
    def remove_filter(self, index):
        """Remove a filter"""
        self.experiment.remove_filter(index)
        self.update_filters_display()
    
    def preview_filters(self):
        """Preview the effect of current filters"""
        if not hasattr(self.experiment, 'active_filters') or not self.experiment.active_filters:
            self.show_message("No Filters", "No active filters to preview.", "info")
            return
        
        try:
            # Apply filters without modifying original data
            filtered_df = self.experiment.filter_data(apply_to_original=False)
            
            original_count = len(self.experiment.regionprops)
            filtered_count = len(filtered_df)
            percentage = filtered_count / original_count * 100
            
            self.show_message(
                "Filter Preview", 
                f"Filters would reduce data from {original_count:,} to {filtered_count:,} rows ({percentage:.1f}%)",
                "info"
            )
            
        except Exception as e:
            self.show_message("Error", f"Error previewing filters: {e}", "error")
    
    def apply_filters(self):
        """Apply all active filters to the data"""
        if not hasattr(self.experiment, 'active_filters') or not self.experiment.active_filters:
            self.show_message("No Filters", "No active filters to apply.", "info")
            return
        
        try:
            # Apply filters to original data
            self.experiment.filter_data(apply_to_original=True)
            
            # Update displays
            self.update_available_columns()
            self.update_callback()
            
            self.show_message("Success", "Filters applied successfully! Data has been updated.", "check")
            
        except Exception as e:
            self.show_message("Error", f"Error applying filters: {e}", "error")
    
    def reset_filters(self):
        """Reset data to original unfiltered state"""
        try:
            self.experiment.reset_data()
            self.experiment.clear_filters()
            
            # Update displays
            self.update_available_columns()
            self.update_filters_display()
            self.update_callback()
            
            self.show_message("Success", "Data reset to original unfiltered state.", "check")
            
        except Exception as e:
            self.show_message("Error", f"Error resetting data: {e}", "error")

class StatsTabFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=15)
        
        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📈 Statistical Analysis Results", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(10, 15))

        # Main tests section
        tests_label = ctk.CTkLabel(
            self, 
            text="Overall Tests", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        tests_label.pack(pady=(10, 5))

        self.stat_tree = ttk.Treeview(
            self, columns=("Test", "Statistic", "p-value"), 
            show="headings", height=5
        )
        self.stat_tree.heading("Test", text="Test")
        self.stat_tree.heading("Statistic", text="Statistic")
        self.stat_tree.heading("p-value", text="p-value")
        
        # Style the treeview
        self.stat_tree.column("Test", width=150)
        self.stat_tree.column("Statistic", width=120)
        self.stat_tree.column("p-value", width=120)
        
        self.stat_tree.pack(fill="x", expand=False, padx=10, pady=5)

        # Post-hoc tests section
        posthoc_label = ctk.CTkLabel(
            self, 
            text="Post-hoc Comparisons", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        posthoc_label.pack(pady=(15, 5))

        self.posthoc_tree = ttk.Treeview(
            self, columns=("group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"), 
            show="headings", height=10
        )
        
        headers = {
            "group1": "Group 1",
            "group2": "Group 2", 
            "meandiff": "Mean Diff",
            "p-adj": "p-adj",
            "lower": "Lower CI",
            "upper": "Upper CI",
            "reject": "Significant"
        }
        
        for col, header in headers.items():
            self.posthoc_tree.heading(col, text=header)
            self.posthoc_tree.column(col, width=100)
        
        self.posthoc_tree.pack(fill="both", expand=True, padx=10, pady=5)

    def update_stats(self, experiment, metric):
        # Clear existing data
        for row in self.stat_tree.get_children():
            self.stat_tree.delete(row)
        for row in self.posthoc_tree.get_children():
            self.posthoc_tree.delete(row)
        
        # Update main tests
        if hasattr(experiment, "anova_results") and experiment.anova_results is not None:
            self.stat_tree.insert("", "end", values=(
                "ANOVA", 
                f"{experiment.anova_results.statistic:.4g}", 
                f"{experiment.anova_results.pvalue:.4g}"
            ))
        
        if hasattr(experiment, "kruskal_results") and experiment.kruskal_results is not None:
            self.stat_tree.insert("", "end", values=(
                "Kruskal-Wallis", 
                f"{experiment.kruskal_results.statistic:.4g}", 
                f"{experiment.kruskal_results.pvalue:.4g}"
            ))
        
        # Update post-hoc tests
        if hasattr(experiment, "tukey_results") and hasattr(experiment.tukey_results, "empty") and not experiment.tukey_results.empty:
            for _, row in experiment.tukey_results.iterrows():
                self.posthoc_tree.insert("", "end", values=tuple(row))

class RegionpropsApp:
    def __init__(self):
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        self.app = ctk.CTk()
        self.app.title("🌭 dashUND Region Properties Analysis")
        self.app.geometry("1600x1000")
        self.app.minsize(1400, 800)
        
        # Set up proper cleanup on window close
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.experiment = Experiment()
        self.plots = None  # Will hold ExperimentPlots instance
        self.group_frames = []  # Initialize empty list
        self.add_another_button = None  # Initialize button reference
        
        # Store canvas references for proper cleanup
        self.active_canvases = []
        self.active_figures = []
        
        # Configure main grid with better proportions
        self.app.grid_columnconfigure(0, weight=0, minsize=600)  # Fixed minimum width for left panel
        self.app.grid_columnconfigure(1, weight=1)  # Plot area expands
        self.app.grid_rowconfigure(0, weight=1)
        
        # Left panel - Controls with tabs (fixed width)
        self.left_panel = ctk.CTkFrame(self.app, width=600)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")  # Changed to "ns" to maintain width
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_propagate(False)  # Prevent shrinking
    
        # Right panel - Plots
        self.right_panel = ctk.CTkFrame(self.app)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)
        
        self.setup_left_panel()
        self.setup_right_panel()
    
    def on_closing(self):
        """Handle application closing with proper cleanup"""
        try:
            # Clear all matplotlib figures and canvases
            self.cleanup_matplotlib()
            
            # Clear any remaining references
            if self.plots:
                self.plots = None
            
            # Force garbage collection
            gc.collect()
            
            # Destroy the main window
            self.app.quit()
            self.app.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Force exit if cleanup fails
            try:
                self.app.destroy()
            except:
                pass
    
    def cleanup_matplotlib(self):
        """Clean up matplotlib figures and canvases"""
        try:
            # Close all stored canvases
            for canvas in self.active_canvases:
                try:
                    if canvas and hasattr(canvas, 'get_tk_widget'):
                        widget = canvas.get_tk_widget()
                        if widget and widget.winfo_exists():
                            widget.destroy()
                except:
                    pass
            
            # Close all stored figures
            for fig in self.active_figures:
                try:
                    if fig:
                        plt.close(fig)
                except:
                    pass
            
            # Clear the lists
            self.active_canvases.clear()
            self.active_figures.clear()
            
            # Close any remaining matplotlib figures
            plt.close('all')
            
        except Exception as e:
            print(f"Error cleaning up matplotlib: {e}")
    
    def clear_frame(self, frame):
        """Clear all widgets from a frame and clean up matplotlib objects"""
        try:
            for widget in frame.winfo_children():
                # If it's a canvas widget, handle it specially
                if hasattr(widget, 'figure'):
                    try:
                        fig = widget.figure
                        if fig in self.active_figures:
                            self.active_figures.remove(fig)
                        plt.close(fig)
                    except:
                        pass
                
                # Remove from active canvases if present
                if widget in self.active_canvases:
                    self.active_canvases.remove(widget)
                
                # Destroy the widget
                widget.destroy()
        except Exception as e:
            print(f"Error clearing frame: {e}")
    
    def create_and_embed_plot(self, plot_frame, figure):
        """Safely create and embed a matplotlib plot"""
        try:
            # Clear the frame first
            self.clear_frame(plot_frame)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(figure, master=plot_frame)
            canvas.draw()
            
            # Store references for cleanup
            self.active_canvases.append(canvas)
            self.active_figures.append(figure)
            
            # Embed the canvas
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            
            return canvas
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            # Create error label instead
            error_label = ctk.CTkLabel(
                plot_frame, 
                text=f"Error generating plot: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.grid(row=0, column=0, padx=5, pady=5)
            return None
    
    def setup_left_panel(self):
        """Setup the left control panel with tabs"""
        # Title for the entire left panel
        title_label = ctk.CTkLabel(
            self.left_panel, 
            text="🌭 dashUND Region Properties Analysis", 
            font=ctk.CTkFont(size=18, weight="bold"),
            wraplength=580
        )
        title_label.grid(row=0, column=0, pady=(10, 10))
        
        # Create tabview for left panel
        self.left_tabs = ctk.CTkTabview(self.left_panel)
        self.left_tabs.add("📂 Data Input")
        self.left_tabs.add("🔍 Filtering")
        self.left_tabs.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure tab frames
        for tab_name in ["📂 Data Input", "🔍 Filtering"]:
            self.left_tabs.tab(tab_name).grid_columnconfigure(0, weight=1)
            self.left_tabs.tab(tab_name).grid_rowconfigure(0, weight=1)
        
        # Setup Data Input tab
        self.setup_data_input_tab()
        
        # Setup Filtering tab
        self.setup_filtering_tab()
    
    def setup_data_input_tab(self):
        """Setup the data input tab with groups and finalize"""
        # Create a scrollable frame for the data input content
        data_input_frame = ctk.CTkScrollableFrame(
            self.left_tabs.tab("📂 Data Input")
        )
        data_input_frame._scrollbar.configure(height=5)  # Shorter scrollbar
        data_input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        data_input_frame.grid_columnconfigure(0, weight=1)
        
        # Instructions with better text wrapping
        instructions = ctk.CTkLabel(
            data_input_frame,
            text="1. Add experimental groups by selecting regionprops CSV files\n"
                 "2. Use the Filtering tab to clean your data (optional)\n"
                 "3. Click 'Finalize & Analyze' to generate visualizations\n"
                 "4. Explore plots using the tabs on the right",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            justify="left",
            wraplength=520  # Adjusted for smaller tab width
        )
        instructions.grid(row=0, column=0, pady=(0, 15), padx=5, sticky="ew")
        
        # Groups section
        groups_label = ctk.CTkLabel(
            data_input_frame, 
            text="📊 Experimental Groups", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        groups_label.grid(row=1, column=0, pady=(10, 10))
        
        # Groups container with responsive sizing
        self.groups_container = ctk.CTkFrame(data_input_frame, fg_color="transparent")
        self.groups_container.grid(row=2, column=0, sticky="ew", padx=5)
        self.groups_container.grid_columnconfigure(0, weight=1)
        
        # Initialize group frames list and add first group frame
        self.group_frames = []
        self.add_group_frame()
        
        # Data summary section
        summary_label = ctk.CTkLabel(
            data_input_frame, 
            text="📈 Data Summary", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        summary_label.grid(row=3, column=0, pady=(20, 8))
        
        # Summary info label with responsive sizing
        self.summary_info = ctk.CTkLabel(
            data_input_frame,
            text="No data loaded yet",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            justify="left",
            wraplength=520,
            anchor="w"
        )
        self.summary_info.grid(row=4, column=0, pady=5, padx=5, sticky="ew")
        
        # Finalize button with responsive width
        self.finalize_button = ctk.CTkButton(
            data_input_frame, 
            text="🚀 Finalize & Analyze Data", 
            command=self.finalize,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45
        )
        self.finalize_button.grid(row=5, column=0, pady=20, padx=5, sticky="ew")

    def setup_filtering_tab(self):
        """Setup the filtering tab"""
        # Create the filter controls frame
        self.filter_controls = FilterControlsFrame(
            self.left_tabs.tab("🔍 Filtering"),
            self.experiment,
            self.update_axis_dropdowns
        )
        self.filter_controls.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    def setup_right_panel(self):
        """Setup the right plot panel"""
        # Plot tabs
        self.plot_tabs = ctk.CTkTabview(self.right_panel)
        self.plot_tabs.add("📊 Categorical")
        self.plot_tabs.add("🔍 Scatter")
        self.plot_tabs.add("🌟 Pairplot")
        self.plot_tabs.add("📈 PCA")
        self.plot_tabs.add("⏰ Time Series")  # Add this new tab
        self.plot_tabs.add("📊 Statistics")
        self.plot_tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure tab frames
        for tab_name in ["📊 Categorical", "🔍 Scatter", "🌟 Pairplot", "📈 PCA", "⏰ Time Series", "📊 Statistics"]:
            self.plot_tabs.tab(tab_name).grid_columnconfigure(0, weight=1)
            self.plot_tabs.tab(tab_name).grid_rowconfigure(1, weight=1)
        
        # Categorical plot setup
        self.cat_controls = CategoricalControlsFrame(
            self.plot_tabs.tab("📊 Categorical"), 
            self.experiment, 
            None,
            self.update_cat_plot
        )
        self.cat_controls.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.cat_plot_frame = ctk.CTkFrame(
            self.plot_tabs.tab("📊 Categorical"), 
            corner_radius=15
        )
        self.cat_plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.cat_plot_frame.grid_columnconfigure(0, weight=1)
        self.cat_plot_frame.grid_rowconfigure(0, weight=1)
        
        # Scatter plot setup
        self.scatter_controls = ScatterControlsFrame(
            self.plot_tabs.tab("🔍 Scatter"), 
            self.experiment, 
            None,
            self.update_scatter_plot
        )
        self.scatter_controls.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.scatter_plot_frame = ctk.CTkFrame(
            self.plot_tabs.tab("🔍 Scatter"), 
            corner_radius=15
        )
        self.scatter_plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.scatter_plot_frame.grid_columnconfigure(0, weight=1)
        self.scatter_plot_frame.grid_rowconfigure(0, weight=1)
        
        # Pairplot setup
        self.pairplot_controls = PairplotControlsFrame(
            self.plot_tabs.tab("🌟 Pairplot"), 
            self.experiment, 
            None,
            self.update_pairplot
        )
        self.pairplot_controls.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.pairplot_frame = ctk.CTkFrame(
            self.plot_tabs.tab("🌟 Pairplot"), 
            corner_radius=15
        )
        self.pairplot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.pairplot_frame.grid_columnconfigure(0, weight=1)
        self.pairplot_frame.grid_rowconfigure(0, weight=1)
        
        # PCA setup
        self.pca_controls = PCAControlsFrame(
            self.plot_tabs.tab("📈 PCA"), 
            self.experiment, 
            None,
            self.update_pca_plot
        )
        self.pca_controls.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.pca_plot_frame = ctk.CTkFrame(
            self.plot_tabs.tab("📈 PCA"), 
            corner_radius=15
        )
        self.pca_plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.pca_plot_frame.grid_columnconfigure(0, weight=1)
        self.pca_plot_frame.grid_rowconfigure(0, weight=1)
        
        # Time Series setup - ADD THIS SECTION
        self.timeseries_controls = TimeSeriesControlsFrame(
            self.plot_tabs.tab("⏰ Time Series"), 
            self.experiment, 
            None,
            self.update_timeseries_plot
        )
        self.timeseries_controls.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.timeseries_plot_frame = ctk.CTkFrame(
            self.plot_tabs.tab("⏰ Time Series"), 
            corner_radius=15
        )
        self.timeseries_plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.timeseries_plot_frame.grid_columnconfigure(0, weight=1)
        self.timeseries_plot_frame.grid_rowconfigure(0, weight=1)
        
        # Statistics setup (keep existing)
        self.stats_tab = StatsTabFrame(self.plot_tabs.tab("📊 Statistics"))
        self.stats_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def add_group_frame(self):
        """Add a new group configuration frame"""
        # Remove existing "Add Another Group" button if it exists
        for widget in self.groups_container.winfo_children():
            if isinstance(widget, ctk.CTkButton) and "Add Another Group" in widget.cget("text"):
                widget.destroy()

        group_frame = GroupFrame(
            self.groups_container, 
            self.experiment, 
            self.update_axis_dropdowns,
            self.remove_group_frame
        )
        group_frame.grid(row=len(self.group_frames), column=0, sticky="ew", pady=5)
        self.group_frames.append(group_frame)
        
        # Add button for next group at the end
        self.add_another_button = ctk.CTkButton(
            self.groups_container,
            text="➕ Add Another Group",
            command=self.add_group_frame,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="transparent",
            border_width=2
        )
        self.add_another_button.grid(row=len(self.group_frames), column=0, pady=8, sticky="ew")
    
    def remove_group_frame(self, frame_to_remove):
        """Remove a group frame"""
        if frame_to_remove in self.group_frames:
            self.group_frames.remove(frame_to_remove)
            frame_to_remove.destroy()
            
            # Reorganize remaining frames
            for i, frame in enumerate(self.group_frames):
                frame.grid(row=i, column=0, sticky="ew", pady=5)
            
            # Update the "Add Another Group" button position
            if hasattr(self, 'add_another_button') and self.add_another_button.winfo_exists():
                self.add_another_button.grid(row=len(self.group_frames), column=0, pady=8, sticky="ew")

    def update_data_summary(self):
        """Update the data summary information"""
        if self.experiment.regionprops.empty:
            self.summary_info.configure(text="No data loaded yet")
        else:
            n_rows = len(self.experiment.regionprops)
            n_groups = self.experiment.regionprops['group'].nunique()
            n_samples = self.experiment.regionprops['sample'].nunique()
            
            # Get group breakdown
            group_counts = self.experiment.regionprops.groupby('group').size()
            group_summary = "\n".join([f"  • {group}: {count:,} cells" for group, count in group_counts.items()])
            
            summary_text = (
                f"📊 Total: {n_rows:,} cells across {n_groups} groups ({n_samples} samples)\n\n"
                f"Group breakdown:\n{group_summary}\n\n"
                f"Available metrics: {len(self.experiment.regionprops.select_dtypes(include=[np.number]).columns)} numeric columns"
            )
            
            self.summary_info.configure(text=summary_text)
    
    def update_axis_dropdowns(self):
        """Update dropdown options when data changes"""
        if not self.experiment.regionprops.empty:
            numeric_cols = self.experiment.regionprops.select_dtypes(include=[np.number]).columns.tolist()
            # Remove unwanted columns
            exclude_cols = ["group", "label", "index", "sample", "frame", "scene",
                          "bbox-0", "bbox-1", "bbox-2", "bbox-3", 
                          "centroid-0", "centroid-1", "track_id"]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            self.scatter_controls.update_dropdowns(numeric_cols)
            self.cat_controls.update_dropdowns(numeric_cols)
            self.pairplot_controls.update_columns(numeric_cols)
            self.timeseries_controls.update_metrics(numeric_cols)  # Add this line
            
            # Update filter controls
            if hasattr(self, 'filter_controls'):
                self.filter_controls.update_available_columns()
            
            # Update data summary
            self.update_data_summary()
            
            # Initialize plots object
            self.plots = ExperimentPlots(
                experiment=self.experiment,
                regionprops=self.experiment.regionprops,
                style="darkgrid",  # Better for dark theme
                palette="Set2"
            )
            
            # Share plots object with control frames
            self.cat_controls.plots = self.plots
            self.scatter_controls.plots = self.plots
            self.pairplot_controls.plots = self.plots
            self.pca_controls.plots = self.plots
            self.timeseries_controls.plots = self.plots  # Add this line
    
    def clear_frame(self, frame):
        """Clear all widgets from a frame"""
        for widget in frame.winfo_children():
            widget.destroy()
    
    def style_all_legends(self, fig):
        """Comprehensively style all legends in a figure for dark theme"""
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
        
        # Style all found legends
        for legend in legends_to_style:
            try:
                frame = legend.get_frame()
                frame.set_facecolor('#2b2b2b')
                frame.set_edgecolor('#ffffff')
                frame.set_alpha(0.9)
                
                for text in legend.get_texts():
                    text.set_color('#ffffff')
                
                if hasattr(legend, 'get_title') and legend.get_title():
                    legend.get_title().set_color('#ffffff')
                    
            except Exception as e:
                print(f"Could not style legend: {e}")
                continue

    def apply_figure_style(self, fig, grid_style='--', grid_alpha=0.3):
        """Apply consistent dark styling to matplotlib figures"""
        # Set figure size to fit better in frames
        fig.set_size_inches(10, 6)
        fig.patch.set_facecolor('#212121')  # Dark background
        
        for ax in fig.axes:
            ax.set_facecolor('#2b2b2b')  # Slightly lighter for plot area
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ffffff')
            ax.spines['bottom'].set_color('#ffffff')
            ax.tick_params(colors='#ffffff', labelsize=9)
            ax.xaxis.label.set_color('#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')
            ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color='#ffffff')
        
        # Style all legends comprehensively
        self.style_all_legends(fig)
        
        # Adjust layout to fit better
        fig.tight_layout(pad=1.0)
    
    def update_scatter_plot(self):
        """Update the scatter plot"""
        if not self.plots:
            return
            
        xaxis = self.scatter_controls.xaxis_var.get()
        yaxis = self.scatter_controls.yaxis_var.get()
        kind = self.scatter_controls.plot_kind_var.get()
        
        if not xaxis or not yaxis:
            return
        
        xlabel = self.scatter_controls.xlabel.get() or xaxis.replace('_', ' ').title()
        ylabel = self.scatter_controls.ylabel.get() or yaxis.replace('_', ' ').title()
        title = self.scatter_controls.title.get() or f'{ylabel} vs {xlabel}'
        
        try:
            fig = self.plots.plot_jointplot(
                x=xaxis, y=yaxis, hue='group', kind=kind,
                title=title, xlabel=xlabel, ylabel=ylabel, show=False
            )
            self.apply_figure_style(fig)
            
            # Use safe plot creation
            self.create_and_embed_plot(self.scatter_plot_frame, fig)
            
        except Exception as e:
            self.clear_frame(self.scatter_plot_frame)
            error_label = ctk.CTkLabel(
                self.scatter_plot_frame, 
                text=f"Error generating plot: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.grid(row=0, column=0)
    
    def update_cat_plot(self):
        """Update the categorical plot"""
        if not self.plots:
            return
            
        plot_type = self.cat_controls.plotType_var.get()
        metric = self.cat_controls.metric_var.get()
        
        if not plot_type or not metric:
            return
        
        title = self.cat_controls.title.get() or f'{metric.replace("_", " ").title()} by Group'
        xlabel = self.cat_controls.xlabel.get() or 'Group'
        ylabel = self.cat_controls.ylabel.get() or metric.replace("_", " ").title()
        annotate = self.cat_controls.annotate_var.get()
        
        try:
            fig = self.plots.plot_categorical_comparisons(
                metric=metric, plot_kind=plot_type, 
                title=title, xlabel=xlabel, ylabel=ylabel,
                annotate=annotate, show=False
            )
            self.apply_figure_style(fig)
            
            # Use safe plot creation
            self.create_and_embed_plot(self.cat_plot_frame, fig)
            
        except Exception as e:
            self.clear_frame(self.cat_plot_frame)
            error_label = ctk.CTkLabel(
                self.cat_plot_frame, 
                text=f"Error generating plot: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.grid(row=0, column=0)
    
    def update_pairplot(self):
        """Update the pairplot"""
        if not self.plots:
            return
        
        selected_columns = self.pairplot_controls.get_selected_columns()
        if not selected_columns:
            return
        
        title = self.pairplot_controls.title.get() or "Pairwise Relationships"
        corner = self.pairplot_controls.corner_var.get()
        
        try:
            height = float(self.pairplot_controls.height_var.get())
        except ValueError:
            height = 2.0  # Smaller default for better fitting
        
        try:
            fig = self.plots.plot_pairplot(
                columns=selected_columns, hue="group", 
                title=title, corner=corner, height=height, show=False
            )
            self.apply_figure_style(fig)
            
            # Use safe plot creation
            self.create_and_embed_plot(self.pairplot_frame, fig)
            
        except Exception as e:
            self.clear_frame(self.pairplot_frame)
            error_label = ctk.CTkLabel(
                self.pairplot_frame, 
                text=f"Error generating plot: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.grid(row=0, column=0)
    
    def update_pca_plot(self):
        """Update the PCA plot"""
        if not self.plots:
            return
        
        try:
            n_components = int(self.pca_controls.n_components_var.get())
        except ValueError:
            n_components = 2
        
        title = self.pca_controls.title.get()
        
        try:
            fig = self.plots.plot_pca(
                n_components=n_components, title=title, show=False
            )
            self.apply_figure_style(fig)
            
            # Use safe plot creation
            self.create_and_embed_plot(self.pca_plot_frame, fig)
            
        except Exception as e:
            self.clear_frame(self.pca_plot_frame)
            error_label = ctk.CTkLabel(
                self.pca_plot_frame, 
                text=f"Error generating plot: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.grid(row=0, column=0)
    
    def update_timeseries_plot(self):
        """Update the time series plot"""
        if not self.plots:
            return
        
        x_col = self.timeseries_controls.x_col_var.get()
        selected_metrics = self.timeseries_controls.get_selected_metrics()
        
        if not x_col or not selected_metrics:
            return
        
        # Check if time column exists
        if x_col not in self.experiment.regionprops.columns:
            # Show info message
            self.clear_frame(self.timeseries_plot_frame)
            info_label = ctk.CTkLabel(
                self.timeseries_plot_frame, 
                text=f"⚠️ Time column '{x_col}' not found in data.\nAvailable columns: {list(self.experiment.regionprops.columns)}",
                font=ctk.CTkFont(size=12),
                text_color="orange",
                justify="center"
            )
            info_label.grid(row=0, column=0, padx=5, pady=5)
            return
        
        # Get plot parameters
        aggregation = self.timeseries_controls.aggregation_var.get()
        error_bars = self.timeseries_controls.error_bars_var.get()
        show_individual = self.timeseries_controls.show_individual_var.get()
        smooth = self.timeseries_controls.smooth_var.get()
        
        try:
            smooth_window = int(self.timeseries_controls.smooth_window_var.get())
        except ValueError:
            smooth_window = 3
        
        title = self.timeseries_controls.title.get() or f'Time Series: {", ".join(selected_metrics[:3])}'
        xlabel = self.timeseries_controls.xlabel.get() or x_col.replace('_', ' ').title()
        ylabel = self.timeseries_controls.ylabel.get() or 'Metric Value'
        
        try:
            fig = self.plots.plot_time_series(
                x_col=x_col, y_cols=selected_metrics,
                title=title, xlabel=xlabel, ylabel=ylabel,
                aggregation=aggregation, error_bars=error_bars,
                show_individual=show_individual, smooth=smooth,
                smooth_window=smooth_window, show=False
            )
            self.apply_figure_style(fig)
            
            # Use safe plot creation
            self.create_and_embed_plot(self.timeseries_plot_frame, fig)
            
        except Exception as e:
            self.clear_frame(self.timeseries_plot_frame)
            error_label = ctk.CTkLabel(
                self.timeseries_plot_frame, 
                text=f"Error generating plot: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.grid(row=0, column=0)

    def finalize(self):
        """Finalize experiment and generate all plots"""
        if self.experiment.regionprops.empty:
            self.show_message("No Data", "Please add at least one group with data files.", "warning")
            return
        
        try:
            # Initialize the experiment
            self.experiment.summarize()
            
            # Update axis dropdowns and initialize plots
            self.update_axis_dropdowns()
            
            # Generate initial plots
            if self.cat_controls.metric_var.get():
                self.update_cat_plot()
                
                # Run statistical comparisons
                metric = self.cat_controls.metric_var.get()
                self.experiment.compare_groups(metric=metric)
                self.stats_tab.update_stats(self.experiment, metric)
            
            if (self.scatter_controls.xaxis_var.get() and 
                self.scatter_controls.yaxis_var.get()):
                self.update_scatter_plot()
            
            # Switch to plots view
            self.plot_tabs.set("📊 Categorical")
            
            # Show success message
            self.show_message("Success", "Analysis completed! Explore the different plot tabs for visualizations.", "check")
            
        except Exception as e:
            self.show_message("Error", f"Error during analysis: {str(e)}", "error")
    
    def show_message(self, title, message, icon="info"):
        """Show message using available method"""
        try:
            # Try CTkMessagebox first (newer versions)
            if hasattr(ctk, 'CTkMessagebox'):
                return ctk.CTkMessagebox(title=title, message=message, icon=icon).get()
            else:
                # Fall back to tkinter messagebox
                if icon == "warning":
                    return messagebox.showwarning(title, message)
                elif icon == "cancel" or icon == "error":
                    return messagebox.showerror(title, message)
                elif icon == "check":
                    return messagebox.showinfo(title, message)
                else:
                    return messagebox.showinfo(title, message)
        except:
            # Ultimate fallback - print to console
            print(f"{title}: {message}")
    
    def run(self):
        """Start the application"""
        try:
            self.app.mainloop()
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Ensure cleanup happens even if mainloop fails
            try:
                self.cleanup_matplotlib()
            except:
                pass

if __name__ == "__main__":
    app = RegionpropsApp()
    app.run()