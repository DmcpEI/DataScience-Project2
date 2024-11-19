import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart, get_variable_types


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

    def plot_single_variable_boxplots(self):
        """Plots boxplots for each numerical variable."""

    def plot_histograms(self):
        """Plots histograms for all numerical variables."""

    def plot_outliers(self):
        """Plots boxplots for outlier detection."""

    def plot_class_distribution(self):
        """Plots the distribution of the target variable."""

    def plot_granularity_analysis(self):
        """Analyzes the number of unique values per variable."""

    def plot_sparsity_analysis(self):
        """Visualizes sparsity in the dataset."""

    def plot_correlation_analysis(self):
        """Displays a correlation heatmap for numerical variables."""