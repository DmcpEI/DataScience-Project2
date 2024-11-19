import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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
        self.dataset = self.data_loader.data
        self.target = self.data_loader.target

    def plot_basic_statistics(self):
        """Plots number of records and variables."""
        records = self.dataset.shape[0]
        variables = self.dataset.shape[1]

        # Plot the bars
        bars = plt.bar(['Records', 'Variables'], [records, variables], color=['skyblue', 'salmon'])

        # Annotate the bars with their values
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of bar)
                height,  # Y-coordinate (top of bar)
                f"{int(height)}",  # Text to display
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=12  # Font size
            )

        # Add labels and title
        plt.title('Number of Records and Variables')
        plt.ylabel('Count')
        plt.show()

    def plot_variable_types(self):
        """Plots the distribution of variable types."""
        types = self.dataset.dtypes.value_counts()

        types.plot(kind='bar', color='skyblue')
        plt.title('Variable Types')
        plt.ylabel('Count')
        plt.xlabel('Data Types')
        plt.show()

    def plot_missing_values(self):
        """Plots the number of missing values in the dataset."""
        # Calculate missing values and filter only those with missing values
        missing_values = self.dataset.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

        if missing_values.empty:
            print("No missing values found in the dataset.")
            return

        # Plot the missing values
        plt.barh(missing_values.index, missing_values.values, color='skyblue')
        plt.title('Missing Values')
        plt.xlabel('Count')
        plt.ylabel('Variables')

        # Annotate the bars with missing values count
        for i, value in enumerate(missing_values):
            plt.text(value, i, str(value), va='center', ha='left', fontsize=10)

        plt.show()

    def plot_global_boxplots(self):
        """Plots boxplots for all numerical variables."""
        numerical_vars = self.dataset.select_dtypes(include='number')

        numerical_vars.boxplot(figsize=(15, 10))
        plt.title('Global Boxplots')
        plt.ylabel('Value')
        plt.show()

    def plot_single_variable_boxplots(self):
        """Plots boxplots for each numerical variable."""
        numerical_vars = self.dataset.select_dtypes(include='number')

        numerical_vars.plot(kind='box', subplots=True, layout=(5, 5), figsize=(15, 15), sharex=False, sharey=False)
        plt.suptitle('Single Variable Boxplots', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def plot_histograms(self):
        """Plots histograms for all numerical variables."""
        numerical_vars = self.dataset.select_dtypes(include='number')

        numerical_vars.hist(bins=20, figsize=(15, 10), color='skyblue')
        plt.suptitle('Histograms', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_outliers(self):
        """Plots boxplots for outlier detection."""
        numerical_vars = self.dataset.select_dtypes(include='number')

        numerical_vars.boxplot(figsize=(15, 10))
        plt.title('Outlier Analysis')
        plt.ylabel('Value')
        plt.show()

    def plot_class_distribution(self):
        """Plots the distribution of the target variable."""
        if self.target in self.dataset.columns:
            self.dataset[self.target].value_counts().plot(kind='bar', color='skyblue')
            plt.title('Class Distribution')
            plt.ylabel('Count')
            plt.xlabel('Classes')
            plt.show()
        else:
            print(f"Target column '{self.target}' not found in the dataset.")

    def plot_granularity_analysis(self):
        """Analyzes the number of unique values per variable."""
        unique_counts = self.dataset.nunique().sort_values()

        unique_counts.plot(kind='bar', color='skyblue')
        plt.title('Granularity Analysis')
        plt.ylabel('Number of Unique Values')
        plt.xlabel('Variables')
        plt.show()

    def plot_sparsity_analysis(self):
        """Visualizes sparsity in the dataset."""
        sns.heatmap(self.dataset.isnull(), cbar=False, cmap='viridis')
        plt.title('Sparsity Analysis')
        plt.show()

    def plot_correlation_analysis(self):
        """Displays a correlation heatmap for numerical variables."""
        correlation_matrix = self.dataset.corr()

        plt.figure(figsize=(15, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()