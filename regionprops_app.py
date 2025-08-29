import customtkinter as ctk
from tkinter import StringVar, BooleanVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilenames
import matplotlib
import os
import numpy as np

from experiment import Experiment, ExperimentSample

class ScatterControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master)
        self.experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback

        self.xaxis_var = StringVar()
        self.yaxis_var = StringVar()
        ctk.CTkLabel(self, text="X axis:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.xaxis_dropdown = ctk.CTkComboBox(self, variable=self.xaxis_var, values=[], width=200)
        self.xaxis_dropdown.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(self, text="Y axis:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.yaxis_dropdown = ctk.CTkComboBox(self, variable=self.yaxis_var, values=[], width=200)
        self.yaxis_dropdown.grid(row=0, column=3, padx=5, pady=5)
        self.update_button = ctk.CTkButton(self, text="Update scatterplot", command=self.update_callback)
        self.update_button.grid(row=0, column=4, padx=5, pady=5)
        self.save_button = ctk.CTkButton(self, text="Save scatterplot", command=self.save_scatterplot)
        self.save_button.grid(row=0, column=5, padx=5, pady=5)
        # Add entries with labels for table title, xaxis label and yaxis label
        self.title = StringVar()
        self.xlabel = StringVar()
        self.ylabel = StringVar()
        ctk.CTkLabel(self, text="Title: ").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.title_entry = ctk.CTkEntry(self, textvariable=self.title, width=200)
        self.title_entry.grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkLabel(self, text="X axis label: ").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.xlabel_entry = ctk.CTkEntry(self, textvariable=self.xlabel, width=200)
        self.xlabel_entry.grid(row=1, column=3, padx=5, pady=5)
        ctk.CTkLabel(self, text="Y axis label: ").grid(row=1, column=4, padx=5, pady=5, sticky="e")
        self.ylabel_entry = ctk.CTkEntry(self, textvariable=self.ylabel, width=200)
        self.ylabel_entry.grid(row=1, column=5, padx=5, pady=5)


    def update_dropdowns(self, numeric_cols):
        self.xaxis_dropdown.configure(values=numeric_cols)
        self.yaxis_dropdown.configure(values=numeric_cols)
        if numeric_cols:
            if not self.xaxis_var.get():
                self.xaxis_var.set(numeric_cols[0])
            if not self.yaxis_var.get() and len(numeric_cols) > 1:
                self.yaxis_var.set(numeric_cols[1])

    def save_scatterplot(self):
        filename = asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename and self.experiment.scatter_plot:
            self.experiment.scatter_plot.savefig(filename)

class CatControlsFrame(ctk.CTkFrame):
    def __init__(self, master, experiment, plot_frame, update_callback):
        super().__init__(master)
        self.experiment = experiment
        self.plot_frame = plot_frame
        self.update_callback = update_callback

        self.plotType_var = StringVar()
        self.metric_var = StringVar()
        ctk.CTkLabel(self, text="Plot type:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.plotType_dropdown = ctk.CTkComboBox(self, variable=self.plotType_var, values=["box", "violin", "boxen", "strip"], width=200)
        self.plotType_dropdown.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(self, text="Metric to compare by:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.metric_dropdown = ctk.CTkComboBox(self, variable=self.metric_var, values=[], width=200)
        self.metric_dropdown.grid(row=0, column=3, padx=5, pady=5)
        self.update_button = ctk.CTkButton(self, text="Update plot", command=self.update_callback)
        self.update_button.grid(row=0, column=4, padx=5, pady=5)
        self.save_button = ctk.CTkButton(self, text="Save plot", command=self.save_cat_plot)
        self.save_button.grid(row=0, column=5, padx=5, pady=5)
        # Add entries with labels for table title, xaxis label and yaxis label
        self.title = StringVar()
        self.xlabel = StringVar()
        self.ylabel = StringVar()
        ctk.CTkLabel(self, text="Title: ").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.title_entry = ctk.CTkEntry(self, textvariable=self.title, width=200)
        self.title_entry.grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkLabel(self, text="X axis label: ").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.xlabel_entry = ctk.CTkEntry(self, textvariable=self.xlabel, width=200)
        self.xlabel_entry.grid(row=1, column=3, padx=5, pady=5)
        ctk.CTkLabel(self, text="Y axis label: ").grid(row=1, column=4, padx=5, pady=5, sticky="e")
        self.ylabel_entry = ctk.CTkEntry(self, textvariable=self.ylabel, width=200)
        self.ylabel_entry.grid(row=1, column=5, padx=5, pady=5)

    def update_dropdowns(self, numeric_cols):
        self.metric_dropdown.configure(values=numeric_cols)
        self.plotType_dropdown.configure(values=["box", "violin", "boxen", "strip", "point", "count", "bar"])
        if numeric_cols:
            if not self.metric_var.get():
                self.metric_var.set(numeric_cols[0])

    def save_cat_plot(self):
        filename = asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename and self.experiment.cat_plot:
            self.experiment.cat_plot.savefig(filename)

class StatsTabFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.stat_tree = ttk.Treeview(self, columns=("Test", "Statistic", "p-value"), show="headings", height=5)
        self.stat_tree.heading("Test", text="Test")
        self.stat_tree.heading("Statistic", text="Statistic")
        self.stat_tree.heading("p-value", text="p-value")
        self.stat_tree.pack(fill="x", expand=False)
        self.posthoc_tree = ttk.Treeview(self, columns=("group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"), show="headings", height=10)
        for col in ("group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"):
            self.posthoc_tree.heading(col, text=col)
        self.posthoc_tree.pack(fill="both", expand=True)

    def update_stats(self, experiment, metric):
        for row in self.stat_tree.get_children():
            self.stat_tree.delete(row)
        for row in self.posthoc_tree.get_children():
            self.posthoc_tree.delete(row)
        if hasattr(experiment, "anova_results") and experiment.anova_results is not None:
            self.stat_tree.insert("", "end", values=("ANOVA", f"{experiment.anova_results.statistic:.4g}", f"{experiment.anova_results.pvalue:.4g}"))
        if hasattr(experiment, "kruskal_results") and experiment.kruskal_results is not None:
            self.stat_tree.insert("", "end", values=("Kruskal-Wallis", f"{experiment.kruskal_results.statistic:.4g}", f"{experiment.kruskal_results.pvalue:.4g}"))
        if hasattr(experiment, "tukey_results") and hasattr(experiment.tukey_results, "empty") and not experiment.tukey_results.empty:
            for _, row in experiment.tukey_results.iterrows():
                self.posthoc_tree.insert("", "end", values=tuple(row))

class RegionpropsApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Dashund regionprops")
        self.app.geometry("1000x700")
        self.experiment = Experiment()
        self.group_frames = []

        # Main and plot frames
        self.main_frame = ctk.CTkScrollableFrame(self.app)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10, side="left")
        self.plot_frame = ctk.CTkFrame(self.app)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10, side="right")

        # Tabs
        self.plot_tabs = ctk.CTkTabview(self.plot_frame)
        self.plot_tabs.add("Statistical comparisons")
        self.plot_tabs.add("Categorical comparisons")
        self.plot_tabs.add("Population scatterplots")

        self.plot_tabs.pack(fill="both", expand=True)
        self.stats_tab = StatsTabFrame(self.plot_tabs.tab("Statistical comparisons"))
        self.stats_tab.pack(fill="both", expand=True, padx=10, pady=10)
        self.cat_plot_frame = ctk.CTkFrame(self.plot_tabs.tab("Categorical comparisons"))
        self.cat_plot_frame.pack(fill="both", expand=True)
        self.cat_plot_frame.rowconfigure(0, weight=1)
        self.cat_plot_frame.columnconfigure(0, weight=1)
        self.scatter_plot_frame = ctk.CTkFrame(self.plot_tabs.tab("Population scatterplots"))
        self.scatter_plot_frame.pack(fill="both", expand=True)
        self.scatter_plot_frame.rowconfigure(0, weight=1)
        self.scatter_plot_frame.columnconfigure(0, weight=1)

        # Controls
        self.scatter_controls = ScatterControlsFrame(self.plot_tabs.tab("Population scatterplots"), self.experiment, self.scatter_plot_frame, self.update_scatter_plot)
        self.scatter_controls.pack(padx=10, pady=10, fill="x")
        self.cat_controls = CatControlsFrame(self.plot_tabs.tab("Categorical comparisons"), self.experiment, self.cat_plot_frame, self.update_cat_plot)
        self.cat_controls.pack(padx=10, pady=10, fill="x")

        # Group controls
        self.add_group_frame()

        # Finalize button
        self.finalize_button = ctk.CTkButton(self.main_frame, text="Finalize & Plot", command=self.finalize)
        self.finalize_button.pack(padx=10, pady=10, side="bottom")

    def add_group_frame(self):
        group_frame = ctk.CTkFrame(self.main_frame)
        group_frame.pack(fill="x", pady=10, padx=10, anchor="n")
        ctk.CTkLabel(group_frame, text="Group name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        group_name_var = StringVar()
        group_name_entry = ctk.CTkEntry(group_frame, textvariable=group_name_var)
        group_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        regionprops_paths = []
        check_vars = []
        checkboxes = []

        def update_add_group_state(*args):
            if group_name_var.get().strip() and any(var.get() for var in check_vars):
                add_group_button.configure(state="normal")
                remove_group_button.configure(state="normal")
            else:
                add_group_button.configure(state="disabled")

        def browse_regionprops():
            for cb in checkboxes:
                cb.destroy()
            checkboxes.clear()
            check_vars.clear()
            regionprops_paths.clear()
            paths = askopenfilenames(title="Select regionprops files for this group")
            for i, path in enumerate(paths):
                var = BooleanVar(value=True)
                check_vars.append(var)
                regionprops_paths.append(path)
                filename = os.path.basename(path)
                cb = ctk.CTkCheckBox(group_frame, text=filename, variable=var, command=update_add_group_state)
                cb.grid(row=2+i, column=0, columnspan=2, sticky="w", padx=20, pady=5)
                checkboxes.append(cb)
            update_add_group_state()

        browse_button = ctk.CTkButton(group_frame, text="Browse regionprops", command=browse_regionprops)
        browse_button.grid(row=0, column=2, columnspan=2, padx=5, pady=5, sticky="w")
        add_group_button = ctk.CTkButton(group_frame, text="Add group", state="disabled")
        add_group_button.grid(row=99, column=0, columnspan=2, pady=5)
        remove_group_button = ctk.CTkButton(group_frame, text="Remove group", state="disabled")
        remove_group_button.grid(row=99, column=2, columnspan=2, pady=5)

        def add_group():
            group = group_name_var.get().strip()
            selected_paths = [p for p, v in zip(regionprops_paths, check_vars) if v.get()]
            for path in selected_paths:
                sample_name = os.path.basename(path).replace('_regionprops.csv', '')
                sample = ExperimentSample(name=sample_name, group=group, regionprops_df_path=path, bitDepth=16, normalize=True)
                self.experiment.add_sample(sample)
            group_name_entry.configure(state="disabled")
            browse_button.configure(state="disabled")
            add_group_button.configure(state="disabled")
            for cb in checkboxes:
                cb.configure(state="disabled")
            self.update_axis_dropdowns()
            self.add_group_frame()
        def remove_group():
            group_name = group_name_var.get().strip()
            self.group_frames.remove(group_frame)
            group_frame.destroy()
            self.experiment.remove_group(group_name)

        add_group_button.configure(command=add_group)
        group_name_var.trace_add("write", update_add_group_state)
        remove_group_button.configure(command=remove_group)
        self.group_frames.append(group_frame)

    def update_axis_dropdowns(self):
        if not self.experiment.regionprops.empty:
            numeric_cols = self.experiment.regionprops.select_dtypes(include=[np.number]).columns.tolist()
            self.scatter_controls.update_dropdowns(numeric_cols)
            self.cat_controls.update_dropdowns(numeric_cols)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def apply_figure_style(self, fig, grid_style='--', grid_alpha=0.7):
        fig.patch.set_facecolor('#f0f0f0')
        axes_bg = '#222222' if matplotlib.rcParams['axes.facecolor'] == '#222222' else matplotlib.rcParams['axes.facecolor']
        for ax in fig.axes:
            ax.set_facecolor(axes_bg)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.grid(True, linestyle=grid_style, alpha=grid_alpha)
            ax.grid(True, linestyle='--', alpha=0.7)

    def update_scatter_plot(self):
        xaxis = self.scatter_controls.xaxis_var.get()
        yaxis = self.scatter_controls.yaxis_var.get()
        xlabel = self.scatter_controls.xlabel_entry.get() if self.scatter_controls.xlabel_entry.get() else xaxis.replace('_', ' ').title()
        ylabel = self.scatter_controls.ylabel_entry.get() if self.scatter_controls.ylabel_entry.get() else yaxis.replace('_', ' ').title()
        title = self.scatter_controls.title_entry.get() if self.scatter_controls.title_entry.get() else f'{yaxis.replace("_", " ").title()} vs {xaxis.replace("_", " ").title()}'
        if xaxis and yaxis:
            self.clear_frame(self.scatter_plot_frame)
            self.experiment.plot_population(x=xaxis, y=yaxis, hue='group', kind='scatter', show=False, title=title, xlabel=xlabel, ylabel=ylabel)
            self.apply_figure_style(self.experiment.scatter_plot, grid_style='--', grid_alpha=0.7)
            canvas1 = FigureCanvasTkAgg(self.experiment.scatter_plot, master=self.scatter_plot_frame)
            canvas1.draw()
            canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def update_cat_plot(self):
        plot_type = self.cat_controls.plotType_var.get()
        metric = self.cat_controls.metric_var.get()
        title = self.cat_controls.title_entry.get() if self.cat_controls.title_entry.get() else f'{metric.replace("_", " ").title()} by Group'
        xlabel = self.cat_controls.xlabel_entry.get() if self.cat_controls.xlabel_entry.get() else 'Group'
        ylabel = self.cat_controls.ylabel_entry.get() if self.cat_controls.ylabel_entry.get() else metric.replace("_", " ").title()
        if plot_type and metric:
            self.clear_frame(self.cat_plot_frame)
            self.experiment.plot_categorical_comparisons(plot_kind=plot_type, metric=metric, show=False, title=title, xlabel=xlabel, ylabel=ylabel)
            self.apply_figure_style(self.experiment.cat_plot, grid_style='--', grid_alpha=0.7)
            canvas2 = FigureCanvasTkAgg(self.experiment.cat_plot, master=self.cat_plot_frame)
            canvas2.draw()
            canvas2.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def finalize(self):
        self.experiment.summarize()
        self.update_cat_plot()
        self.update_scatter_plot()
        metric = self.cat_controls.metric_var.get()
        self.experiment.compare_groups(metric=metric)
        self.stats_tab.update_stats(self.experiment, metric)

    def run(self):
        self.app.mainloop()

    def on_exit(self):
        self.app.quit()
        self.app.destroy()



if __name__ == "__main__":
    RegionpropsApp().run()