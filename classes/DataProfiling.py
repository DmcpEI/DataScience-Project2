import math

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, savefig, show
from numpy import ndarray
import numpy as np
import pandas as pd
from pandas import Series
from scipy.stats import norm, expon, lognorm

from dslabs_functions import plot_bar_chart, get_variable_types, define_grid, HEIGHT, plot_multiline_chart, \
    set_chart_labels


class DataVisualization:
    """
    A class for visualizing data using various plot types.

    Parameters:
        data_loader (DataLoader): A DataLoader object that provides access to the data.

    Methods:
        plot_basic_statistics(): Plots basic statistics like number of records and variables.
        plot_variable_types(): Plots the distribution of variable types.
        plot_missing_values(): Visualizes missing values in the dataset.
        plot_global_boxplots(): Plots global boxplots for numerical variables.
        plot_single_variable_boxplots(): Plots individual boxplots for numerical variables.
        plot_histograms(): Plots histograms for numerical variables.
        plot_outliers(): Visualizes outliers using boxplots.
        plot_class_distribution(): Plots the class distribution.
        plot_granularity_analysis(): Analyzes unique values per variable.
        plot_sparsity_analysis(): Visualizes sparsity in the dataset.
        plot_correlation_analysis(): Displays a correlation heatmap.
    """

    def __init__(self, data_loader):
        """
        Initializes the DataVisualization class with a DataLoader object.

        Parameters:
            data_loader (DataLoader): The DataLoader object used to load the data.
        """
        self.data_loader = data_loader
        self.data = self.data_loader.data
        self.target = self.data_loader.target

    def plot_records_variables(self):
        """Plots number of records and variables."""
        figure(figsize=(6, 5))
        values: dict[str, int] = {"Nº records": self.data.shape[0], "Nº variables": self.data.shape[1]}
        plot_bar_chart(
            list(values.keys()), list(values.values()), title="Nº of records vs Nº variables"
        )
        savefig(f"graphs/{self.data_loader.file_tag}_records_variables.png")
        show()

    def plot_variable_types(self):
        """Plots the distribution of variable types."""
        variable_types: dict[str, list] = get_variable_types(self.data)
        print(variable_types)
        counts: dict[str, int] = {}
        for tp in variable_types.keys():
            counts[tp] = len(variable_types[tp])

        figure(figsize=(6, 5))
        plot_bar_chart(
            list(counts.keys()), list(counts.values()), title="Nº of variables per type"
        )
        savefig(f"graphs/{self.data_loader.file_tag}_variable_types.png")
        show()

    def plot_missing_values(self):
        """Plots the number of missing values in the dataset."""
        mv: dict[str, int] = {}
        for var in self.data.columns:
            nr: int = self.data[var].isna().sum()
            if nr > 0:
                mv[var] = nr

        figure(figsize=(9, 8))
        plot_bar_chart(
            list(mv.keys()),
            list(mv.values()),
            title="Nr of missing values per variable",
            xlabel="variables",
            ylabel="nr missing values",
        )
        savefig(f"graphs/{self.data_loader.file_tag}_mv.png")
        show()

    def plot_global_boxplots(self):
        """Plots boxplots for all numerical variables."""
        variables_types: dict[str, list] = get_variable_types(self.data)
        numeric: list[str] = variables_types["numeric"]
        if [] != numeric:
            figure(figsize=(8, 7))
            self.data[numeric].boxplot(rot=45)
            savefig(f"graphs/{self.data_loader.file_tag}_global_boxplot.png")
            show()
        else:
            print("There are no numeric variables.")

    def plot_single_variable_boxplots(self):
        """Plots boxplots for each numerical variable."""
        variables_types: dict[str, list] = get_variable_types(self.data)
        numeric: list[str] = variables_types["numeric"]
        numeric = [col for col in numeric if pd.api.types.is_numeric_dtype(self.data[col])]
        if numeric:
            # Determine the size of the grid
            num_plots = len(numeric)
            grid_size = math.ceil(math.sqrt(num_plots))  # Square grid (equal rows and columns)

            # Create the figure and axes with a square grid
            fig: Figure
            axs: ndarray
            fig, axs = plt.subplots(
                grid_size, grid_size, figsize=(grid_size * HEIGHT, grid_size * HEIGHT), squeeze=False
            )

            i, j = 0, 0
            for n in range(len(numeric)):
                axs[i, j].set_title(f"Boxplot for {numeric[n]}")
                axs[i, j].boxplot(self.data[numeric[n]].dropna().values)
                i, j = (i + 1, 0) if (n + 1) % grid_size == 0 else (i, j + 1)

            # Remove any empty subplots
            for ax in axs.flat[num_plots:]:
                ax.axis('off')

            # Save and display the plot
            savefig(f"graphs/{self.data_loader.file_tag}_single_boxplots.png")
            show()
        else:
            print("There are no numeric variables.")

    def plot_histograms(self):
        """Plots histograms for all numerical variables."""
        variables_types: dict[str, list] = get_variable_types(self.data)
        numeric: list[str] = variables_types["numeric"]
        numeric = [col for col in numeric if pd.api.types.is_numeric_dtype(self.data[col])]

        if numeric:
            # Determine the size of the grid
            num_plots = len(numeric)
            grid_size = math.ceil(math.sqrt(num_plots))  # Square grid (equal rows and columns)

            # Create the figure and axes with a square grid
            fig: Figure
            axs: ndarray
            fig, axs = plt.subplots(
                grid_size, grid_size, figsize=(grid_size * HEIGHT, grid_size * HEIGHT), squeeze=False
            )
            i: int
            j: int
            i, j = 0, 0

            for n in range(len(numeric)):
                feature = numeric[n]
                col_data = self.data[feature].dropna()

                # Calculate value range for dynamic bin adjustment
                value_range = col_data.max() - col_data.min()
                bins = max(10, min(100, int(value_range / 1000)))  # Dynamic bins: adjust divisor (1000) as needed

                print(f"Processing feature: {feature} | Range: {value_range} | Bins: {bins}")

                # Set labels and plot histogram
                set_chart_labels(
                    axs[i, j],
                    title=f"Histogram for {feature}",
                    xlabel=feature,
                    ylabel="nr records",
                )
                axs[i, j].hist(col_data.values, bins=bins)  # Use dynamic bins here

                # Update grid indices
                i, j = (i + 1, 0) if (n + 1) % grid_size == 0 else (i, j + 1)

            savefig(f"graphs/{self.data_loader.file_tag}_histogram_numeric_distribution.png")
            plt.show()
        else:
            print("There are no numeric variables.")

        symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
        if [] != symbolic:
            rows, cols = define_grid(len(symbolic))
            fig, axs = plt.subplots(
                rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
            )
            i, j = 0, 0
            for n in range(len(symbolic)):
                counts: Series = self.data[symbolic[n]].value_counts()
                plot_bar_chart(
                    counts.index.to_list(),
                    counts.to_list(),
                    ax=axs[i, j],
                    title="Histogram for %s" % symbolic[n],
                    xlabel=symbolic[n],
                    ylabel="nr records",
                    percentage=False,
                )
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            savefig(f"graphs/{self.data_loader.file_tag}_histograms_symbolic.png")
            show()
        else:
            print("There are no symbolic variables.")
    NR_STDEV: int = 2
    IQR_FACTOR: float = 1.5
    HEIGHT = 6  # Adjust as per your grid size and visualization needs

    @staticmethod
    def determine_outlier_thresholds_for_var(
            summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
    ) -> tuple[float, float]:
        """
        Determine outlier thresholds for a variable based on IQR or standard deviation.
        """
        top: float = 0
        bottom: float = 0
        if std_based:
            std: float = threshold * summary5["std"]
            top = summary5["mean"] + std
            bottom = summary5["mean"] - std
        else:
            iqr: float = threshold * (summary5["75%"] - summary5["25%"])
            top = summary5["75%"] + iqr
            bottom = summary5["25%"] - iqr

        return top, bottom

    @staticmethod
    def count_outliers(
            data: pd.DataFrame,
            numeric: list[str],
            nrstdev: int = NR_STDEV,
            iqrfactor: float = IQR_FACTOR,
    ) -> dict:
        """
        Count outliers for numerical variables based on IQR and stdev criteria.
        """
        outliers_iqr: list = []
        outliers_stdev: list = []
        summary5: pd.DataFrame = data[numeric].describe()

        for var in numeric:
            # Standard deviation-based thresholds
            top, bottom = DataVisualization.determine_outlier_thresholds_for_var(
                summary5[var], std_based=True, threshold=nrstdev
            )
            outliers_stdev.append(
                (data[var] > top).sum() + (data[var] < bottom).sum()
            )

            # IQR-based thresholds
            top, bottom = DataVisualization.determine_outlier_thresholds_for_var(
                summary5[var], std_based=False, threshold=iqrfactor
            )
            outliers_iqr.append(
                (data[var] > top).sum() + (data[var] < bottom).sum()
            )

        return {"iqr": outliers_iqr, "stdev": outliers_stdev}

    def plot_outlier_comparison(self):
        """
        Plot a comparison of outlier counts using IQR and stdev criteria.
        """
        numeric: list[str] = get_variable_types(self.data)["numeric"]
        numeric = [col for col in numeric if pd.api.types.is_numeric_dtype(self.data[col])]
        if numeric:
            # Count outliers
            outliers: dict[str, list] = self.count_outliers(self.data, numeric)

            # Create a multibar chart
            figure(figsize=(12, self.HEIGHT))
            x = np.arange(len(numeric))
            width = 0.35

            # Bar plots for IQR and stdev
            plt.bar(x - width / 2, outliers["iqr"], width, label="IQR")
            plt.bar(x + width / 2, outliers["stdev"], width, label="Stdev")

            # Add labels and title
            plt.title("Comparison of Outliers per Variable")
            plt.xlabel("Variables")
            plt.ylabel("Number of Outliers")
            plt.xticks(ticks=x, labels=numeric, rotation=45, ha="right")
            plt.legend()

            # Save and show the plot
            savefig(f"graphs/{self.data_loader.file_tag}_outliers_comparison.png")
            show()
        else:
            print("There are no numeric variables.")

    def plot_class_distribution(self, target):
        """Plots the distribution of the target variable."""
        values: Series = self.data[target].value_counts()
        print(values)

        figure(figsize=(4, 2))
        plot_bar_chart(
            values.index.to_list(),
            values.to_list(),
            title=f"Target distribution (target={target})",
        )
        savefig(f"graphs/{self.data_loader.file_tag}_class_distribution.png")
        show()

    def plot_granularity_analysis(self):
        """Analyzes the number of unique values per variable."""

    def plot_sparsity_analysis(self):
        """Visualizes sparsity in the dataset."""

    def plot_correlation_analysis(self):
        """Displays a correlation heatmap for numerical variables."""