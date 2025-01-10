# %% Class Imports
import json

from classes.DataLoader import DataManipulator
from classes.DataProfiling import DataProfiling
from classes.DataProcessing import DataProcessing
from classes.DataModeling import DataModeling
from config.dslabs_functions import dataframe_temporal_train_test_split

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
# data_profiling1.augmented_dicker_fuller_test()
# data_profiling2.augmented_dicker_fuller_test()

# %% 2- Data Preparation

# Data Processing
data_processing1 = DataProcessing(data_loader1)
data_processing2 = DataProcessing(data_loader2)

# Handle Missing Values
data_processing1.apply_best_missing_value_approach_forecasting('Mean', {'Mean': 0})

data_processing2.apply_best_missing_value_approach_forecasting('Mean', {'Mean': 0})

# Save the data
data_loader1.data.to_csv("data/forecast_ny_arrests_MV.csv", index=True)
data_loader2.data.to_csv("data/forecast_gdp_europe_MV.csv", index=True)
# data_loader1.data = pd.read_csv("data/forecast_ny_arrests_MV.csv", index_col="Date")
# data_loader2.data = pd.read_csv("data/forecast_gdp_europe_MV.csv", index_col="Year")

# Handle Scaling
techniques1 = data_processing1.handle_scaling_forecasting()
print(f"\nFrom the plots we conclude that the best approach is to scale the {data_loader1.file_tag} dataset\n")
data_processing1.apply_best_scaling_approach_forecasting('Scaled', techniques1)

techniques2 = data_processing2.handle_scaling_forecasting()
print(f"\nFrom the plots we conclude that the best approach is to scale the {data_loader2.file_tag} dataset\n")
data_processing2.apply_best_scaling_approach_forecasting('Scaled', techniques2)

# Save the data
data_loader1.data.to_csv("data/forecast_ny_arrests_scaled.csv", index=True)
data_loader2.data.to_csv("data/forecast_gdp_europe_scaled.csv", index=True)
# data_loader1.data = pd.read_csv("data/forecast_ny_arrests_scaled.csv", index_col="Date")
# data_loader2.data = pd.read_csv("data/forecast_gdp_europe_scaled.csv", index_col="Year")

# Handle Aggregation
techniques1 = data_processing1.handle_aggregation_forecasting()
print(f"\nFrom the plots we conclude that the best approach is to aggregate the {data_loader1.file_tag} dataset by month\n")
data_processing1.apply_best_aggregation_approach_forecasting("monthly", techniques1)

techniques2 = data_processing2.handle_aggregation_forecasting()
print(f"\nFrom the plots we conclude that the best approach is to not aggregate the {data_loader2.file_tag} dataset\n")
data_processing2.apply_best_aggregation_approach_forecasting("Original", techniques2)

# Save the data
data_loader1.data.to_csv("data/forecast_ny_arrests_aggregated.csv", index=True)
data_loader2.data.to_csv("data/forecast_gdp_europe_aggregated.csv", index=True)
# data_loader1.data = pd.read_csv("data/forecast_ny_arrests_aggregated.csv", index_col="Date")
# data_loader2.data = pd.read_csv("data/forecast_gdp_europe_aggregated.csv", index_col="Year")

# Prepare training and testing features/targets
train1, test1 = dataframe_temporal_train_test_split(data_loader1.data, trn_pct=0.90)
X1_train = train1.drop(data_loader1.target, axis=1)
y1_train = train1[data_loader1.target]
X1_test = test1.drop(data_loader1.target, axis=1)
y1_test = test1[data_loader1.target]

train2, test2 = dataframe_temporal_train_test_split(data_loader2.data, trn_pct=0.90)
X2_train = train2.drop(data_loader2.target, axis=1)
y2_train = train2[data_loader2.target]
X2_test = test2.drop(data_loader2.target, axis=1)
y2_test = test2[data_loader2.target]

# Save the data in the data processing class
data_processing1.X_train = X1_train
data_processing1.X_test = X1_test
data_processing1.y_train = y1_train
data_processing1.y_test = y1_test

data_processing2.X_train = X2_train
data_processing2.X_test = X2_test
data_processing2.y_train = y2_train
data_processing2.y_test = y2_test

# Handle Smoothing
techniques1, smooth_data1 = data_processing1.handle_smoothing_forecasting(train1, X1_train, X1_test, y1_train, y1_test)
print(f"\nFrom the plots we conclude that the best approach is to smooth the {data_loader1.file_tag} "
      f"dataset with a smoothing window size of \n")
data_processing1.apply_best_smoothing_approach_forecasting("Original", techniques1, smooth_data1)

techniques2, smooth_data2 = data_processing2.handle_smoothing_forecasting(train2, X2_train, X2_test, y2_train, y2_test)
print(f"\nFrom the plots we conclude that the best approach is to smooth the {data_loader2.file_tag} "
        f"dataset with a smoothing window size of \n")
data_processing2.apply_best_smoothing_approach_forecasting("Original", techniques2, smooth_data2)

X1_train, y1_train = data_processing1.X_train, data_processing1.y_train
X2_train, y2_train = data_processing2.X_train, data_processing2.y_train

# Save the data
X2_train_without_diff, y2_train_without_diff = X2_train, y2_train
X2_test_without_diff, y2_test_without_diff = X2_test, y2_test

# Handle Differentiation
techniques1, differentiated_data1 = data_processing1.handle_differentiation_forecasting(X1_train, X1_test, y1_train, y1_test)
print(f"\nFrom the plots we conclude that the best approach is to differentiate the {data_loader1.file_tag} dataset\n")
data_processing1.apply_best_differentiation_approach_forecasting("Original", techniques1, differentiated_data1)

techniques2, differentiated_data2 = data_processing2.handle_differentiation_forecasting(X2_train, X2_test, y2_train, y2_test)
print(f"\nFrom the plots we conclude that the best approach is to differentiate the {data_loader2.file_tag} dataset\n")
data_processing2.apply_best_differentiation_approach_forecasting("Second Derivative", techniques2, differentiated_data2)

X1_train, y1_train, X1_test, y1_test = data_processing1.X_train, data_processing1.y_train, data_processing1.X_test, data_processing1.y_test
X2_train, y2_train, X2_test, y2_test = data_processing2.X_train, data_processing2.y_train, data_processing2.X_test, data_processing2.y_test

# Save the data
X1_train.to_csv("data/forecast_ny_arrests_X_train_preparation.csv", index=True)
y1_train.to_csv("data/forecast_ny_arrests_y_train_preparation.csv", index=True)
X1_test.to_csv("data/forecast_ny_arrests_X_test_preparation.csv", index=True)
y1_test.to_csv("data/forecast_ny_arrests_y_test_preparation.csv", index=True)

X2_train.to_csv("data/forecast_gdp_europe_X_train_preparation.csv", index=True)
y2_train.to_csv("data/forecast_gdp_europe_y_train_preparation.csv", index=True)
X2_test.to_csv("data/forecast_gdp_europe_X_test_preparation.csv", index=True)
y2_test.to_csv("data/forecast_gdp_europe_y_test_preparation.csv", index=True)

# %% 3- Data Modeling

# Data Modeling
data_modeling1 = DataModeling(data_loader1, X1_train, X1_test, y1_train, y1_test)
data_modeling2 = DataModeling(data_loader2, X2_train, X2_test, y2_train, y2_test)

# Initialize a dictionary to store results
results_summary = {
    "NY Arrests": {},  # Dataset 1
    "GDP": {}  # Dataset 2
}

# Simple Average
print("\nSimple Average Forecasting Approach:")
results_summary["NY Arrests"]["Simple Average"] = data_modeling1.simple_average_model_forecasting()
results_summary["GDP"]["Simple Average"] = data_modeling2.simple_average_model_forecasting()

print("\nSimple Average Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["Simple Average"])
print("GDP:", results_summary["GDP"]["Simple Average"])

# Persistence Model
print("\nPersistence Model Forecasting Approach:")
results_summary["NY Arrests"]["Persistence Model"] = data_modeling1.persistence_model_forecasting()
results_summary["GDP"]["Persistence Model"] = data_modeling2.persistence_model_forecasting()

print("\nPersistence Model Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["Persistence Model"])
print("GDP:", results_summary["GDP"]["Persistence Model"])

# Rolling Mean
print("\nRolling Mean Forecasting Approach:")
results_summary["NY Arrests"]["Rolling Mean"] = data_modeling1.rolling_mean_model_forecasting()
results_summary["GDP"]["Rolling Mean"] = data_modeling2.rolling_mean_model_forecasting()

print("\nRolling Mean Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["Rolling Mean"])
print("GDP:", results_summary["GDP"]["Rolling Mean"])

# Exponential Smoothing
print("\nExponential Smoothing Forecasting Approach:")
results_summary["NY Arrests"]["Exponential Smoothing"] = data_modeling1.exponential_smoothing_model_forecasting()
results_summary["GDP"]["Exponential Smoothing"] = data_modeling2.exponential_smoothing_model_forecasting()

print("\nExponential Smoothing Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["Exponential Smoothing"])
print("GDP:", results_summary["GDP"]["Exponential Smoothing"])

# Linear Regression
print("\nLinear Regression Forecasting Approach:")
results_summary["NY Arrests"]["Linear Regression"] = data_modeling1.linear_regression_model_forecasting()
results_summary["GDP"]["Linear Regression"] = data_modeling2.linear_regression_model_forecasting()

print("\nLinear Regression Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["Linear Regression"])
print("GDP:", results_summary["GDP"]["Linear Regression"])

# ARIMA Univariate
print("\nARIMA Forecasting Approach:")
results_summary["NY Arrests"]["ARIMA_Univariate"] = data_modeling1.arima_model_forecasting()
results_summary["GDP"]["ARIMA_Univariate"] = data_modeling2.arima_model_forecasting()

print("\nARIMA Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["ARIMA_Univariate"])
print("GDP:", results_summary["GDP"]["ARIMA_Univariate"])

# LSTM Univariate
print("\nLSTM Forecasting Approach:")
results_summary["NY Arrests"]["LSTM_Univariate"] = data_modeling1.lstm_model_forecasting()
results_summary["GDP"]["LSTM_Univariate"] = data_modeling2.lstm_model_forecasting()

print("\nLSTM Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["LSTM_Univariate"])
print("GDP:", results_summary["GDP"]["LSTM_Univariate"])

# Multivariate Forecasting
techniques1, multivariate_data1 = data_processing1.handle_differentiation_multivariate_forecasting(X1_train, X1_test, y1_train, y1_test)
print(f"\nFrom the plots we conclude that the best approach is to use the following features for the {data_loader1.file_tag} dataset\n")
data_processing1.apply_best_differentiation_multivariate_approach_forecasting("Original", techniques1, multivariate_data1)

techniques2, multivariate_data2 = data_processing2.handle_differentiation_multivariate_forecasting(X2_train_without_diff, X2_test_without_diff, y2_train_without_diff, y2_test_without_diff)
print(f"\nFrom the plots we conclude that the best approach is to use the following features for the {data_loader2.file_tag} dataset\n")
data_processing2.apply_best_differentiation_multivariate_approach_forecasting("First Derivative", techniques2, multivariate_data2)

# Save the data
X1_train, y1_train, X1_test, y1_test = data_processing1.X_train, data_processing1.y_train, data_processing1.X_test, data_processing1.y_test
X2_train, y2_train, X2_test, y2_test = data_processing2.X_train, data_processing2.y_train, data_processing2.X_test, data_processing2.y_test

# Save the data
data_modeling1.X_train = X1_train
data_modeling1.X_test = X1_test
data_modeling1.y_train = y1_train
data_modeling1.y_test = y1_test

data_modeling2.X_train = X2_train
data_modeling2.X_test = X2_test
data_modeling2.y_train = y2_train
data_modeling2.y_test = y2_test

# ARIMA Multivariate
print("\nARIMA Multivariate Forecasting Approach:")
results_summary["NY Arrests"]["ARIMA_Multivariate"] = data_modeling1.arima_model_forecasting(multivariate=True)
results_summary["GDP"]["ARIMA_Multivariate"] = data_modeling2.arima_model_forecasting(multivariate=True)

print("\nARIMA Multivariate Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["ARIMA_Multivariate"])
print("GDP:", results_summary["GDP"]["ARIMA_Multivariate"])

# LSTM Multivariate
print("\nLSTM Multivariate Forecasting Approach:")
results_summary["NY Arrests"]["LSTM_Multivariate"] = data_modeling1.lstm_model_forecasting(multivariate=True)
results_summary["GDP"]["LSTM_Multivariate"] = data_modeling2.lstm_model_forecasting(multivariate=True)

print("\nLSTM Multivariate Forecasting Results")
print("NY Arrests:", results_summary["NY Arrests"]["LSTM_Multivariate"])
print("GDP:", results_summary["GDP"]["LSTM_Multivariate"])

# Save results to a file (optional)
with open("forecasting_evaluation_results_summary.json", "w") as file:
    json.dump(results_summary, file, indent=4)

# Print the full summary at the end
print("\nFull Evaluation Results Summary:")
print(json.dumps(results_summary, indent=4))