# %% Class Imports
import pandas as pd

from classes.DataLoader import DataManipulator
from classes.DataProfiling import DataProfiling
from classes.DataProcessing import DataProcessing
from classes.DataModeling import DataModeling

# %% 0- Data Loading

# Read Options for the first dataset
read_options_dataset1 = {
    "index_col": "Date",
    "sep": ";",                    # Use a comma as the delimiter
    "decimal": ".",                # Use a period as the decimal separator
    "parse_dates": True,           # Automatically parse date columns
    "infer_datetime_format": True, # Infer datetime format for faster parsing
}

# Read Options for the second dataset
read_options_dataset2 = {
    "index_col": "Year",
    "sep": ",",                    # Use a comma as the delimiter
    "decimal": ".",                # Use a period as the decimal separator
    "parse_dates": True,           # Automatically parse date columns
    "infer_datetime_format": True, # Infer datetime format for faster parsing
}

# Load the data
path_dataset1 = "data/forecast_ny_arrests.csv"
path_dataset2 = "data/forecast_gdp_europe.csv"

data_loader1 = DataManipulator(path_dataset1, "Manhattan", read_options_dataset1)
data_loader2 = DataManipulator(path_dataset2, "GDP", read_options_dataset2)

# Display the data
print(data_loader1.data.head())
print(data_loader2.data.head())

# %% 1- Data Profiling

# Data Visualization
data_profiling1 = DataProfiling(data_loader1)
data_profiling2 = DataProfiling(data_loader2)

# Data Dimensionality
# data_profiling1.plot_unvariate_forecasting()
# data_profiling2.plot_unvariate_forecasting()

# data_profiling1.plot_multivariate_forecasting()
# data_profiling2.plot_multivariate_forecasting()

# Data Granularity
# data_profiling1.plot_granularity_forecasting()
# data_profiling2.plot_granularity_forecasting()

# Data Distribution
# data_profiling1.plot_distribuition_boxplot()
# data_profiling2.plot_distribuition_boxplot()

# data_profiling1.plot_distribuition_histograms()
# data_profiling2.plot_distribuition_histograms()

# data_profiling1.plot_distribuition_lag_plots()
# data_profiling2.plot_distribuition_lag_plots()

# data_profiling1.plot_autocorrelation()
# data_profiling2.plot_autocorrelation()

# Data Stationarity
# data_profiling1.plot_sesonality()
# data_profiling2.plot_sesonality()
#
# data_profiling1.augmented_dicker_fuller_test()
# data_profiling2.augmented_dicker_fuller_test()