import seaborn as sns
from matplotlib.pyplot import figure, savefig, show, subplots
from dslabs_functions import plot_bar_chart, get_variable_types, derive_date_variables, HEIGHT, \
    plot_multi_scatters_chart
from numpy import ndarray
from pandas import Series, DataFrame
from matplotlib.figure import Figure

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

    def _analyse_date_granularity(self, data: DataFrame, var: str, levels: list[str]) -> ndarray:

        cols: int = len(levels)
        fig: Figure()
        axs: ndarray
        fig, axs = subplots(1, cols, figsize=(2 * cols * HEIGHT, HEIGHT), squeeze=False)
        fig.suptitle(f"Granularity study for {var}")

        for i in range(cols):
            counts: Series[int] = data[var + "_" + levels[i]].value_counts()
            plot_bar_chart(
                counts.index.to_list(),
                counts.to_list(),
                ax=axs[0, i],
                title=levels[i],
                xlabel=levels[i],
                ylabel="nr records",
                percentage=False,
            )
        return axs

    def plot_date_granularity_analysis(self):
        """Analyzes the granularity of date variables."""

        variables_types: dict[str, list] = get_variable_types(self.data)
        data_ext: DataFrame = derive_date_variables(self.data, variables_types["date"])

        for v_date in variables_types["date"]:
            self._analyse_date_granularity(self.data, v_date, ["year", "quarter", "month", "day"])
            savefig(f"graphs/{self.data_loader.file_tag}_granularity_{v_date}.png")
            show()

    def _decompose_lat_lon(self, data: DataFrame) -> DataFrame:
        """
        Decomposes Latitude and Longitude into combined Hemisphere, Degrees, Minutes, and Seconds.

        Parameters:
            data (DataFrame): The dataset containing Latitude and Longitude columns.

        Returns:
            DataFrame: Updated dataset with decomposed location features.
        """
        def convert_to_dms(coord):
            degrees = int(coord)
            minutes = int((abs(coord) - abs(degrees)) * 60)
            seconds = (abs(coord) - abs(degrees) - minutes / 60) * 3600
            return degrees, minutes, seconds

        def hemisphere_combined(lat, lon):
            lat_hemi = "Northern" if lat >= 0 else "Southern"
            lon_hemi = "Eastern" if lon >= 0 else "Western"
            return f"{lat_hemi}-{lon_hemi}"

        # Decompose Latitude and Longitude
        data['Hemisphere'] = data.apply(lambda row: hemisphere_combined(row['Latitude'], row['Longitude']), axis=1)

        data['Latitude Degrees'], data['Latitude Minutes'], data['Latitude Seconds'] = zip(
            *data['Latitude'].apply(lambda x: convert_to_dms(x))
        )
        data['Longitude Degrees'], data['Longitude Minutes'], data['Longitude Seconds'] = zip(
            *data['Longitude'].apply(lambda x: convert_to_dms(x))
        )

        return data

    def _analyse_property_granularity(self, data: DataFrame, property: str, vars: list[str]) -> ndarray:
        """
        Analyzes the granularity of a property by plotting distributions.

        Parameters:
            data (DataFrame): The dataset to analyze.
            property (str): The property name for analysis.
            vars (list[str]): List of variables representing different levels of granularity.

        Returns:
            ndarray: Axes of the generated plots.
        """
        cols: int = len(vars)
        fig, axs = subplots(1, cols, figsize=(cols * 5, 5), squeeze=False)
        fig.suptitle(f"Granularity study for {property}")
        for i in range(cols):
            counts: Series[int] = data[vars[i]].value_counts()
            plot_bar_chart(
                counts.index.to_list(),
                counts.to_list(),
                ax=axs[0, i],
                title=vars[i],
                xlabel=vars[i],
                ylabel="nr records",
                percentage=False,
            )
        return axs

    def plot_location_granularity_analysis(self):
        """
        Analyzes the granularity of location variables.

        Decomposes Latitude and Longitude into combined Hemisphere, Degrees, Minutes, and Seconds,
        and then visualizes their distributions. Skips rows with null Latitude or Longitude.
        """
        # Drop rows with null Latitude or Longitude
        data = self.data.dropna(subset=['Latitude', 'Longitude'])

        if data.empty:
            print("No valid Latitude or Longitude data to analyze.")
            return

        # Decompose Latitude and Longitude into granular levels
        data = self._decompose_lat_lon(data)

        # Analyze granularity of the decomposed location variables
        self._analyse_property_granularity(
            data,
            "location",
            ["Hemisphere", "Latitude Degrees", "Longitude Degrees"]
        )
        savefig(f"graphs/{self.data_loader.file_tag}_granularity_location.png")
        show()

    def plot_sparsity_analysis(self):
        """Visualizes sparsity in the dataset."""

        data = self.data.dropna()

        vars: list = data.columns.to_list()
        if [] != vars:

            n: int = len(vars) - 1
            fig: Figure
            axs: ndarray
            fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
            for i in range(len(vars)):
                var1: str = vars[i]
                for j in range(i + 1, len(vars)):
                    var2: str = vars[j]
                    plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])
            savefig(f"graphs/{self.data_loader.file_tag}_sparsity_study.png")
            show()
        else:
            print("Sparsity class: there are no variables.")

    def plot_correlation_analysis(self):
        """Displays a correlation heatmap for numerical variables."""