# %% Class Imports
from classes.DataLoader import DataManipulator
from classes.DataProfiling import DataProfiling
from classes.DataProcessing import DataProcessing

# %% 0- Data Loading

# Load the data
path_dataset1_class = "data/class_ny_arrests.csv"
path_dataset2_class = "data/class_financial distress.csv"

data_loader1 = DataManipulator(path_dataset1_class, "LAW_CAT_CD")
data_loader2 = DataManipulator(path_dataset2_class, "CLASS")

# Display the data
print(data_loader1.data.head())
print(data_loader2.data.head())

# %% 1- Data Profiling

# Data Visualization
data_profiling1 = DataProfiling(data_loader1)
data_profiling2 = DataProfiling(data_loader2)

# Data Dimensionality
# data_profiling1.plot_records_variables()
# data_profiling2.plot_records_variables()
# data_profiling1.plot_variable_types()
# data_profiling2.plot_variable_types()
# data_profiling1.plot_missing_values()
# data_profiling2.plot_missing_values()

# Data Distribution
# data_profiling1.plot_global_boxplots()
# data_profiling2.plot_global_boxplots()
# data_profiling1.plot_single_variable_boxplots()
# data_profiling2.plot_single_variable_boxplots()
# data_profiling1.plot_histograms()
# data_profiling2.plot_histograms()
# data_profiling1.plot_outlier_comparison()
# data_profiling2.plot_outlier_comparison()
# data_profiling1.plot_class_distribution()
# data_profiling2.plot_class_distribution()

# Data Granularity
# data_profiling1.plot_date_granularity_analysis()
# data_profiling1.plot_location_granularity_analysis()
# data_profiling1.plot_law_code_granularity_analysis()
# data_profiling1.plot_borough_granularity_analysis()
# data_profiling1.plot_age_granularity_analysis()
# data_profiling1.plot_race_granularity_analysis()

# Data Sparsity
# data_profiling1.plot_sparsity_analysis()
# data_profiling2.plot_sparsity_analysis()
# data_profiling1.plot_sparsity_analysis_per_class()
# data_profiling2.plot_sparsity_analysis_per_class()

# Data Encoding of NY Arrests
data_processing1 = DataProcessing(data_loader1)
data_processing1.encode_NY_ARRESTS()
# Save the encoded data
data_loader1.data.to_csv("data/class_ny_arrests_encoded.csv", index=False)

data_profiling1 = DataProfiling(data_loader1)

# Data Correlation
data_profiling1.plot_correlation_analysis()
data_profiling2.plot_correlation_analysis()

# %% 2- Data Processing