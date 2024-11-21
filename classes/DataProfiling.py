import math
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, savefig, show, subplots
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from seaborn import heatmap
from scipy.stats import norm, expon, lognorm
from dslabs_functions import plot_bar_chart, get_variable_types, derive_date_variables, HEIGHT, \
    plot_multi_scatters_chart, set_chart_labels, define_grid

class DataProfiling:
    """
    DataProfiling class for visualizing data profiling results.

    Attributes:
        data_loader (DataLoader): The DataLoader object used to load the data.
        data (DataFrame): The dataset to visualize.
        target (str): The target variable in the dataset.

    Methods:
        plot_records_variables: Plots number of records and variables.
        plot_variable_types: Plots the distribution of variable types.
        plot_missing_values: Plots the number of missing values in the dataset.
        plot_global_boxplots: Plots boxplots for all numerical variables.
        plot_single_variable_boxplots: Plots boxplots for each numerical variable.
        plot_histograms: Plots histograms for all numerical variables.
        plot_outliers: Plots boxplots for outlier detection.
        plot_class_distribution: Plots the distribution of the target variable.
        plot_date_granularity_analysis: Analyzes the granularity of date variables.
        plot_location_granularity_analysis: Analyzes the granularity of location variables.
        plot_sparsity_analysis: Visualizes sparsity in the dataset.
        plot_sparsity_analysis_per_class: Visualizes sparsity in the dataset per class.
        plot_correlation_analysis: Displays a correlation heatmap for numerical variables.
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

            savefig(f"graphs/{self.data_loader.file_tag}_single_histograms_numeric.png")
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
            savefig(f"graphs/{self.data_loader.file_tag}_single_histograms_symbolic.png")
            show()
        else:
            print("There are no symbolic variables.")

    def _compute_known_distributions(self, x_values: list) -> dict:
        distributions = dict()
        # Gaussian
        mean, sigma = norm.fit(x_values)
        distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
        # Exponential
        loc, scale = expon.fit(x_values)
        distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
        # LogNorm
        sigma, loc, scale = lognorm.fit(x_values)
        distributions["LogNor(%.1f,%.2f)" % (math.log(scale), sigma)] = lognorm.pdf(
            x_values, sigma, loc, scale
        )
        return distributions

    def histogram_with_distributions(self, ax: Axes, series: Series, var: str):
        values: list = series.sort_values().to_list()
        ax.hist(values, bins=20, density=True, alpha=0.6, label="Data")
        distributions: dict = self._compute_known_distributions(values)
        for label, dist in distributions.items():
            ax.plot(values, dist, label=label)
        ax.set_title(f"Best fit for {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.legend()

    def plot_histograms_distribution(self):

        """Plots histograms for all numerical variables and fits known distributions."""
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

                print(f"Processing feature: {feature}")

                # Plot histogram and distributions
                self.histogram_with_distributions(axs[i, j], col_data, feature)

                # Update grid indices
                i, j = (i + 1, 0) if (n + 1) % grid_size == 0 else (i, j + 1)

            # Remove any empty subplots
            for ax in axs.flat[num_plots:]:
                ax.axis("off")

            savefig(f"graphs/{self.data_loader.file_tag}_histogram_numeric_distribution.png")
            plt.show()
        else:
            print("There are no numeric variables.")

    NR_STDEV: int = 2
    IQR_FACTOR: float = 1.5
    HEIGHT = 6  # Adjust as per your grid size and visualization needs

    @staticmethod
    def determine_outlier_thresholds_for_var(summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
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
            top, bottom = DataProfiling.determine_outlier_thresholds_for_var(
                summary5[var], std_based=True, threshold=nrstdev
            )
            outliers_stdev.append(
                (data[var] > top).sum() + (data[var] < bottom).sum()
            )

            # IQR-based thresholds
            top, bottom = DataProfiling.determine_outlier_thresholds_for_var(
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

    def plot_class_distribution(self):
        """Plots the distribution of the target variable."""
        values: Series = self.data[self.data_loader.target].value_counts()
        print(values)

        figure(figsize=(4, 2))
        plot_bar_chart(
            values.index.to_list(),
            values.to_list(),
            title=f"Target distribution (target={self.data_loader.target})",
        )
        savefig(f"graphs/{self.data_loader.file_tag}_class_distribution.png")
        show()

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
            ["Hemisphere", "Latitude Degrees", "Longitude Degrees",
             "Latitude Minutes", "Longitude Minutes",
             "Latitude Seconds", "Longitude Seconds"]
        )
        savefig(f"graphs/{self.data_loader.file_tag}_granularity_location.png")
        show()

    def plot_law_code_granularity_analysis(self):
        if self.data.empty:
            print("No valid LAW_CAT_CD data to analyze.")
            return

        self._analyse_property_granularity(
            self.data,
            "law code",
            ["OFNS_DESC", "PD_DESC", "LAW_CODE"]
        )

        savefig(f"graphs/{self.data_loader.file_tag}_granularity_law_code.png")
        show()

    def plot_borough_granularity_analysis(self):
        """
        Analyzes the granularity of race variables.
        """
        # Analyze granularity of the race variable
        self._analyse_property_granularity(
            self.data,
            "borough",
            ["ARREST_BORO"]
        )
        savefig(f"graphs/{self.data_loader.file_tag}_granularity_borough.png")
        show()

    def plot_age_granularity_analysis(self):
        """
        Analyzes the granularity of race variables.
        """
        # Analyze granularity of the race variable
        self._analyse_property_granularity(
            self.data,
            "age",
            ["AGE_GROUP"]
        )
        savefig(f"graphs/{self.data_loader.file_tag}_granularity_age.png")
        show()

    def plot_race_granularity_analysis(self):
        """
        Analyzes the granularity of race variables.
        """
        # Analyze granularity of the race variable
        self._analyse_property_granularity(
            self.data,
            "race",
            ["PERP_RACE"]
        )
        savefig(f"graphs/{self.data_loader.file_tag}_granularity_race.png")
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

    def plot_sparsity_analysis_per_class(self):
        """Visualizes sparsity in the dataset."""

        data = self.data.dropna()
        vars: list = data.columns.to_list()
        if [] != vars:

            n: int = len(vars) - 1
            fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
            for i in range(len(vars)):
                var1: str = vars[i]
                for j in range(i + 1, len(vars)):
                    var2: str = vars[j]
                    plot_multi_scatters_chart(data, var1, var2, self.data_loader.target, ax=axs[i, j - 1])
            savefig(f"graphs/{self.data_loader.file_tag}_sparsity_per_class_study.png")
            show()
        else:
            print("Sparsity per class: there are no variables.")

    def plot_correlation_analysis(self, dpi=300, figsize=(15, 15)):
        """
        Displays a correlation heatmap for all variables in the dataset.

        Parameters:
            dpi (int): Dots per inch for the output image resolution.
            figsize (tuple): Size of the figure in inches (width, height).
        """
        # Compute the correlation matrix for all columns
        corr_mtx: DataFrame = self.data.corr().abs()

        # Increase figure size and resolution
        figure(figsize=figsize, dpi=dpi)
        heatmap(
            corr_mtx,
            xticklabels=corr_mtx.columns,
            yticklabels=corr_mtx.columns,
            annot=False,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        savefig(f"graphs/{self.data_loader.file_tag}_correlation_analysis.png", dpi=dpi)
        show()