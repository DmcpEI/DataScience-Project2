import math
import pandas as pd
from matplotlib.gridspec import GridSpec
from pandas import Series, DataFrame
import numpy as np
from numpy import ndarray, array
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, savefig, show, subplots, plot, legend
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from seaborn import heatmap
from scipy.stats import norm, expon, lognorm
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from config.dslabs_functions import plot_bar_chart, get_variable_types, derive_date_variables, HEIGHT, \
    plot_multi_scatters_chart, set_chart_labels, define_grid, plot_line_chart, plot_multiline_chart


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

    def plot_records_variables_classification(self):
        """Plots number of records and variables."""
        figure(figsize=(6, 5))
        values: dict[str, int] = {"Nº records": self.data.shape[0], "Nº variables": self.data.shape[1]}
        plot_bar_chart(
            list(values.keys()), list(values.values()), title="Nº of records vs Nº variables"
        )
        savefig(f"graphs/classification/data_profiling/data_dimensionality/{self.data_loader.file_tag}_records_variables.png")
        show()

    def plot_variable_types_classification(self):
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
        savefig(f"graphs/classification/data_profiling/data_dimensionality/{self.data_loader.file_tag}_variable_types.png")
        show()

    def plot_missing_values_classification(self):
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
        savefig(f"graphs/classification/data_profiling/data_dimensionality/{self.data_loader.file_tag}_mv.png")
        show()

    def plot_global_boxplots_classification(self):
        """Plots boxplots for all numerical variables."""

        variables_types: dict[str, list] = get_variable_types(self.data)
        numeric: list[str] = variables_types["numeric"]
        if [] != numeric:
            figure(figsize=(8, 7))
            self.data[numeric].boxplot(rot=45)
            savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_global_boxplot.png")
            show()
        else:
            print("There are no numeric variables.")

    def plot_single_variable_boxplots_classification(self):
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
            savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_single_boxplots.png")
            show()
        else:
            print("There are no numeric variables.")

    def plot_histograms_classification(self):
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

            savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_single_histograms_numeric.png")
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
            savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_single_histograms_symbolic.png")
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

    def plot_histograms_distribution_classification(self):

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

            savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_histogram_numeric_distribution.png")
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

    def plot_outlier_comparison_classification(self):
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
            savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_outliers_comparison.png")
            show()
        else:
            print("There are no numeric variables.")

    def plot_class_distribution_classification(self):
        """Plots the distribution of the target variable."""
        values: Series = self.data[self.data_loader.target].value_counts()
        print(values)

        figure(figsize=(4, 2))
        plot_bar_chart(
            values.index.to_list(),
            values.to_list(),
            title=f"Target distribution (target={self.data_loader.target})",
        )
        savefig(f"graphs/classification/data_profiling/data_distribution/{self.data_loader.file_tag}_class_distribution.png")
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

    def plot_date_granularity_analysis_classification(self):
        """Analyzes the granularity of date variables."""

        variables_types: dict[str, list] = get_variable_types(self.data)
        data_ext: DataFrame = derive_date_variables(self.data, variables_types["date"])

        for v_date in variables_types["date"]:
            self._analyse_date_granularity(self.data, v_date, ["year", "quarter", "month", "day"])
            savefig(f"graphs/classification/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_{v_date}.png")
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

    def plot_location_granularity_analysis_classification(self):
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
        savefig(f"graphs/classification/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_location.png")
        show()

    def plot_law_code_granularity_analysis_classification(self):
        if self.data.empty:
            print("No valid LAW_CAT_CD data to analyze.")
            return

        self._analyse_property_granularity(
            self.data,
            "law code",
            ["OFNS_DESC", "PD_DESC", "LAW_CODE"]
        )

        savefig(f"graphs/classification/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_law_code.png")
        show()

    def plot_borough_granularity_analysis_classification(self):
        """
        Analyzes the granularity of race variables.
        """
        # Analyze granularity of the race variable
        self._analyse_property_granularity(
            self.data,
            "borough",
            ["ARREST_BORO"]
        )
        savefig(f"graphs/classification/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_borough.png")
        show()

    def plot_age_granularity_analysis_classification(self):
        """
        Analyzes the granularity of race variables.
        """
        # Analyze granularity of the race variable
        self._analyse_property_granularity(
            self.data,
            "age",
            ["AGE_GROUP"]
        )
        savefig(f"graphs/classification/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_age.png")
        show()

    def plot_race_granularity_analysis_classification(self):
        """
        Analyzes the granularity of race variables.
        """
        # Analyze granularity of the race variable
        self._analyse_property_granularity(
            self.data,
            "race",
            ["PERP_RACE"]
        )
        savefig(f"graphs/classification/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_race.png")
        show()

    def plot_sparsity_analysis_classification(self):
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
            savefig(f"graphs/classification/data_profiling/data_sparsity/{self.data_loader.file_tag}_sparsity_study.png")
            show()
        else:
            print("Sparsity class: there are no variables.")

    def plot_sparsity_analysis_per_class_classification(self):
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
            savefig(f"graphs/classification/data_profiling/data_sparsity/{self.data_loader.file_tag}_sparsity_per_class_study.png")
            show()
        else:
            print("Sparsity per class: there are no variables.")

    def plot_correlation_analysis_classification(self, dpi=300, figsize=(15, 15)):
        """
        Displays a correlation heatmap for all variables in the dataset.

        Parameters:
            dpi (int): Dots per inch for the output image resolution.
            figsize (tuple): Size of the figure in inches (width, height).
        """
        # Compute the correlation matrix for all columns
        corr_mtx: DataFrame = self.data_loader.data.corr().abs()

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
        savefig(f"graphs/classification/data_profiling/data_sparsity/{self.data_loader.file_tag}_correlation_analysis.png", dpi=dpi)
        show()

    # %% Forecasting

    def plot_unvariate_forecasting(self):

        series: Series = self.data_loader.data[self.data_loader.target]
        print("\nNr. Records = ", series.shape[0])
        print("First timestamp", series.index[0])
        print("Last timestamp", series.index[-1])

        if self.data_loader.file_tag == "forecast_ny_arrests":
            original_level = "daily"
        else:
            original_level = "yearly"

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel=series.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} {original_level} {self.data_loader.target}",
        )
        savefig(f"graphs/forecasting/data_profiling/data_dimensionality/{self.data_loader.file_tag}_unvariate_forecasting.png")
        show()

    def _plot_ts_multivariate_chart(self, data: DataFrame, title: str) -> list[Axes]:
        fig: Figure
        axs: list[Axes]
        fig, axs = subplots(data.shape[1], 1, figsize=(3 * HEIGHT, HEIGHT / 2 * data.shape[1]))
        fig.suptitle(title)

        for i in range(data.shape[1]):
            col: str = data.columns[i]
            plot_line_chart(
                data[col].index.to_list(),
                data[col].to_list(),
                ax=axs[i],
                xlabel=data.index.name,
                ylabel=col,
            )
        return axs

    def plot_multivariate_forecasting(self):

        print("\nNr. Records = ", self.data_loader.data.shape)
        print("First timestamp", self.data_loader.data.index[0])
        print("Last timestamp", self.data_loader.data.index[-1])

        self._plot_ts_multivariate_chart(self.data_loader.data, title=f"{self.data_loader.file_tag} {self.data_loader.target}")
        savefig(f"graphs/forecasting/data_profiling/data_dimensionality/{self.data_loader.file_tag}_multivariate_forecasting.png")
        show()

    def ts_aggregation_by(
            self,
            data: Series | DataFrame,
            gran_level: str = "D",  # Default to daily
    ) -> Series | DataFrame:
        df = data.copy()

        # Determine aggregation function based on dataset type
        agg_func = "sum" if self.data_loader.file_tag == "forecast_ny_arrests" else "mean"

        # Check if input is a Series
        is_series = isinstance(df, Series)
        if is_series:
            df = df.to_frame(name="value")  # Convert Series to DataFrame for processing

        if gran_level in ["D", "M", "Y", "Q", "W"]:
            # Standard pandas frequency aliases
            index = df.index.to_period(gran_level)
            df = df.groupby(by=index).agg(agg_func)
            df.index = df.index.to_timestamp()  # Convert back to timestamp
        elif gran_level == "two_years":
            # Custom logic for two years
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df["two_years"] = (df.index.year // 2) * 2
            df = df.groupby("two_years").agg(agg_func)
            df.index = pd.to_datetime(df.index, format="%Y")
        elif gran_level == "five_years":
            # Custom logic for five years
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df["five_years"] = (df.index.year // 5) * 5
            df = df.groupby("five_years").agg(agg_func)
            df.index = pd.to_datetime(df.index, format="%Y")  # Convert five years to timestamp
        else:
            raise ValueError(f"Unsupported granularity level: {gran_level}")

        # Convert back to Series if input was a Series
        if is_series:
            df = df.squeeze()  # Convert single-column DataFrame to Series

        return df

    def plot_granularity_forecasting(self):
        """
        Plots the granularity study for the forecasting target.
        """
        series: Series = self.data_loader.data[self.data_loader.target]

        # Determine the granularity levels based on the file tag
        if self.data_loader.file_tag == "forecast_ny_arrests":
            grans: list[str] = ["D", "M", "Y"]
        else:
            grans: list[str] = ["Y", "two_years", "five_years"]

        # Define the desired width for the plots
        plot_width = 3 * HEIGHT  # Keep the width consistent with the original layout
        plot_height = HEIGHT / 2  # Adjust the height for individual plots

        # Loop through each granularity level and create a separate plot
        for gran in grans:
            ss: Series = self.ts_aggregation_by(series, gran)

            # Create a new figure for each granularity level with consistent width
            fig, ax = subplots(1, 1, figsize=(plot_width, plot_height))
            fig.suptitle(f"{self.data_loader.file_tag} {self.data_loader.target} granularity={gran}")

            # Plot the data
            plot_line_chart(
                ss.index.to_list(),
                ss.to_list(),
                ax=ax,
                xlabel=f"{ss.index.name} ({gran})",
                ylabel=self.data_loader.target,
            )

            # Save each plot with a unique filename
            savefig(
                f"graphs/forecasting/data_profiling/data_granularity/{self.data_loader.file_tag}_granularity_{gran}.png")

            # Show the plot
            show()

    def plot_distribuition_boxplot(self):

        series: Series = self.data_loader.data[self.data_loader.target]

        if self.data_loader.file_tag == "forecast_ny_arrests":
            ss_daily: Series = self.ts_aggregation_by(series, "D")
            ss_monthly: Series = self.ts_aggregation_by(series, "M")
            ss_yearly: Series = self.ts_aggregation_by(series, "Y")

            aggregations = {
                "DAILY": ss_daily,
                "MONTHLY": ss_monthly,
                "YEARLY": ss_yearly
            }
        else:
            # Aggregation for yearly, five-year, and decade
            ss_yearly: Series = self.ts_aggregation_by(series, "Y")
            ss_two_years: Series = self.ts_aggregation_by(series, "two_years")
            ss_five_years: Series = self.ts_aggregation_by(series, "five_years")

            aggregations = {
                "YEARLY": ss_yearly,
                "TWO-YEAR": ss_two_years,
                "FIVE-YEAR": ss_five_years
            }

        # Create subplots based on the number of aggregations
        num_aggregations = len(aggregations)
        fig, axs = subplots(2, num_aggregations, figsize=(2 * HEIGHT, HEIGHT))

        for i, (title, agg_series) in enumerate(aggregations.items()):
            set_chart_labels(axs[0, i], title=title)
            axs[0, i].boxplot(agg_series)

            axs[1, i].grid(False)
            axs[1, i].set_axis_off()
            axs[1, i].text(0.2, 0, str(agg_series.describe()), fontsize="small")

        # Hide any remaining unused subplots
        for j in range(num_aggregations, 3):
            axs[0, j].grid(False)
            axs[0, j].set_axis_off()

        savefig(f"graphs/forecasting/data_profiling/data_distribution/{self.data_loader.file_tag}_distribuition_boxplot.png")
        show()

    def plot_distribuition_histograms(self):
        series: Series = self.data_loader.data[self.data_loader.target]

        if self.data_loader.file_tag == "forecast_ny_arrests":
            # Aggregation for daily, monthly, and yearly
            ss_daily: Series = self.ts_aggregation_by(series, gran_level="D")
            ss_monthly: Series = self.ts_aggregation_by(series, gran_level="M")
            ss_yearly: Series = self.ts_aggregation_by(series, gran_level="Y")

            grans: list[Series] = [ss_daily, ss_monthly, ss_yearly]
            gran_names: list[str] = ["Daily", "Monthly", "Yearly"]
        else:
            # Aggregation for yearly, five-year, and decade
            ss_yearly: Series = self.ts_aggregation_by(series, gran_level="Y")
            ss_two_years: Series = self.ts_aggregation_by(series, gran_level="two_years")
            ss_five_years: Series = self.ts_aggregation_by(series, gran_level="five_years")

            grans: list[Series] = [ss_yearly, ss_two_years, ss_five_years]
            gran_names: list[str] = ["Yearly", "Two-Year", "Five-Year"]

        # Create subplots based on the number of aggregations
        fig: Figure
        axs: array
        fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
        fig.suptitle(f"{self.data_loader.file_tag} {self.data_loader.target}")

        for i in range(len(grans)):
            set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=self.data_loader.target, ylabel="Nr records")
            axs[i].hist(grans[i].values)

        savefig(f"graphs/forecasting/data_profiling/data_distribution/{self.data_loader.file_tag}_distribuition_histograms.png")
        show()

    def _get_lagged_series(self, series: Series, max_lag: int, delta: int = 1):
        lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
        for i in range(delta, max_lag + 1, delta):
            lagged_series[f"lag {i}"] = series.shift(i)
        return lagged_series

    def plot_distribuition_lag_plots(self):

        series: Series = self.data_loader.data[self.data_loader.target]

        figure(figsize=(3 * HEIGHT, HEIGHT))
        lags = self._get_lagged_series(series, 20, 10)
        plot_multiline_chart(series.index.to_list(), lags, xlabel=self.data_loader.read_options["index_col"], ylabel=self.data_loader.target)
        savefig(f"graphs/forecasting/data_profiling/data_distribution/{self.data_loader.file_tag}_distribuition_lag_plots.png")
        show()

    def _autocorrelation_study(self, series: Series, max_lag: int, delta: int = 1):
        k: int = int(max_lag / delta)
        fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
        gs = GridSpec(2, k, figure=fig)

        series_values: list = series.tolist()
        for i in range(1, k + 1):
            ax = fig.add_subplot(gs[0, i - 1])
            lag = i * delta
            ax.scatter(series.shift(lag).tolist(), series_values)
            ax.set_xlabel(f"lag {lag}")
            ax.set_ylabel("original")
        ax = fig.add_subplot(gs[1, :])
        ax.acorr(series, maxlags=max_lag)
        ax.set_title("Autocorrelation")
        ax.set_xlabel("Lags")
        return

    def plot_autocorrelation(self):

        series: Series = self.data_loader.data[self.data_loader.target]
        self._autocorrelation_study(series, 10, 1)
        savefig(f"graphs/forecasting/data_profiling/data_distribution/{self.data_loader.file_tag}_autocorrelation.png")
        show()

    def _plot_components(
            self, series: Series, title: str = "", x_label: str = "time", y_label: str = "",
    ) -> list[Axes]:
        decomposition: DecomposeResult = seasonal_decompose(series, model="add")
        components: dict = {
            "observed": series,
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
        }
        rows: int = len(components)
        fig: Figure
        axs: list[Axes]
        fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
        fig.suptitle(f"{title}")
        i: int = 0
        for key in components:
            set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
            axs[i].plot(components[key])
            i += 1
        return axs

    def plot_sesonality(self):

        series: Series = self.data_loader.data[self.data_loader.target]

        if self.data_loader.file_tag == "forecast_ny_arrests":
            original_level = "Daily"
        else:
            original_level = "Yearly"

        self._plot_components(
            series,
            title=f"{self.data_loader.file_tag} - {self.data_loader.target} ({original_level})",
            x_label=series.index.name,
            y_label=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_profiling/data_stationarity/{self.data_loader.file_tag}_components.png")
        show()

        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel=series.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} Stationary Study",
            name="original",
        )
        n: int = len(series)
        plot(series.index, [series.mean()] * n, "r-", label="mean")
        legend()
        savefig(f"graphs/forecasting/data_profiling/data_stationarity/{self.data_loader.file_tag}_stationarity_study.png")
        show()

        BINS = 10
        mean_line: list[float] = []

        for i in range(BINS):
            segment: Series = series[i * n // BINS: (i + 1) * n // BINS]
            mean_value: list[float] = [segment.mean()] * (n // BINS)
            mean_line += mean_value
        mean_line += [mean_line[-1]] * (n - len(mean_line))

        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel=series.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} Stationary Study with STDev",
            name="original",
            show_stdev=True,
        )
        n: int = len(series)
        plot(series.index, mean_line, "r-", label="mean")
        legend()
        savefig(f"graphs/forecasting/data_profiling/data_stationarity/{self.data_loader.file_tag}_stationarity_study_stdev.png")
        show()

    def augmented_dicker_fuller_test(self):

        def eval_stationarity(series: Series) -> bool:
            result = adfuller(series)
            print(f"ADF Statistic: {result[0]:.3f}")
            print(f"p-value: {result[1]:.3f}")
            print("Critical Values:")
            for key, value in result[4].items():
                print(f"\t{key}: {value:.3f}")
            return result[1] <= 0.05

        print(f"\nAugmented Dickey-Fuller Test for {self.data_loader.target} from the {self.data_loader.file_tag} dataset:")
        series: Series = self.data_loader.data[self.data_loader.target]
        print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary\n")