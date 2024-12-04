# %% Class Imports
import pandas as pd
from sklearn.model_selection import train_test_split

from classes.DataLoader import DataManipulator
from classes.DataProfiling import DataProfiling
from classes.DataProcessing import DataProcessing
from classes.DataModeling import DataModeling

# %% 0- Data Loading

# Load the data
path_dataset1_class = "data/class_ny_arrests.csv"
path_dataset2_class = "data/class_financial distress.csv"

data_loader1 = DataManipulator(path_dataset1_class, "JURISDICTION_CODE")
data_loader2 = DataManipulator(path_dataset2_class, "CLASS")

# Display the data
print(data_loader1.data.head())
print(data_loader2.data.head())

# Sample the data of NY Arrests
print(f"\nSampling 100000 records from the NY Arrests dataset...")
data_loader1.data = data_loader1.data.sample(n=100000)
print(f"Dataset successfully sampled to 100000 records.")

# %% 1- Data Profiling

# Data Visualization
data_profiling1 = DataProfiling(data_loader1)
data_profiling2 = DataProfiling(data_loader2)

# Data Processing
data_processing1 = DataProcessing(data_loader1)
data_processing2 = DataProcessing(data_loader2)

data_processing1.pre_encode_variables()

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
# data_profiling1.plot_histograms_distribution()
# data_profiling2.plot_histograms_distribution()
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
data_processing1.encode_variables()
# Save the encoded data
# data_loader1.data.to_csv("data/class_ny_arrests_encoded.csv", index=False)

# Data Correlation
# data_profiling1.plot_correlation_analysis()
# data_profiling2.plot_correlation_analysis()

# %% 2- Data Processing

# Drop False Predictors
data_processing1.drop_variables()
data_processing2.drop_variables()

# Handle Missing Values
techniques1 = data_processing1.handle_missing_values()
print(f"\nForm the plots we conclude that the best approach for the Missing Values of the {data_loader1.file_tag} dataset is Missing Value Removal\n")
data_processing1.apply_best_missing_value_approach('Remove MV', techniques1)

techniques2 = data_processing2.handle_missing_values()
print(f"\nThe {data_loader2.file_tag} dataset doesnt have missing values\n")

# Save the data
data_loader1.data.to_csv("data/class_ny_arrests_MV.csv", index=False)
data_loader2.data.to_csv("data/class_financial_distress_MV.csv", index=False)
# data_loader1.data = pd.read_csv("data/class_ny_arrests_MV.csv")
# data_loader2.data = pd.read_csv("data/class_financial_distress_MV.csv")

# Data Splitting
X1 = data_loader1.data.drop(columns=[data_loader1.target])
y1 = data_loader1.data[data_loader1.target]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
data_processing1.X_train, data_processing1.y_train, data_processing1.X_test, data_processing1.y_test = X1_train, y1_train, X1_test, y1_test

X2 = data_loader2.data.drop(columns=[data_loader2.target])
y2 = data_loader2.data[data_loader2.target]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)
data_processing2.X_train, data_processing2.y_train, data_processing2.X_test, data_processing2.y_test = X2_train, y2_train, X2_test, y2_test

# Handle Outliers
techniques1, df_train_dropped1, df_train_replaced1, df_train_truncated1 = data_processing1.handle_outliers()
print(f"\nForm the plots we conclude that the best approach for the Outliers of the {data_loader1.file_tag} dataset is to drop them\n")
data_processing1.apply_best_outliers_approach('Drop', techniques1,
                                              df_train_dropped1.drop(columns=[data_loader1.target]),
                                              df_train_dropped1[data_loader1.target])

techniques2, df_train_dropped2, df_train_replaced2, df_train_truncated2 = data_processing2.handle_outliers()
print(f"\nForm the plots we conclude that the best approach for the Outliers of the {data_loader2.file_tag} dataset is to drop them\n")
data_processing2.apply_best_outliers_approach('Drop', techniques2,
                                              df_train_dropped2.drop(columns=[data_loader2.target]),
                                              df_train_dropped2[data_loader2.target])

# Save the data
X1_train, X1_test, y1_train, y1_test = (data_processing1.X_train, data_processing1.X_test,
                                        data_processing1.y_train, data_processing1.y_test)

X2_train, X2_test, y2_train, y2_test = (data_processing2.X_train, data_processing2.X_test,
                                        data_processing2.y_train, data_processing2.y_test)

X1_train.to_csv("data/class_ny_arrests_Xtrain_outliers.csv", index=False)
X1_test.to_csv("data/class_ny_arrests_Xtest_outliers.csv", index=False)
y1_train.to_csv("data/class_ny_arrests_ytrain_outliers.csv", index=False)
y1_test.to_csv("data/class_ny_arrests_ytest_outliers.csv", index=False)

X2_train.to_csv("data/class_financial_distress_Xtrain_outliers.csv", index=False)
X2_test.to_csv("data/class_financial_distress_Xtest_outliers.csv", index=False)
y2_train.to_csv("data/class_financial_distress_ytrain_outliers.csv", index=False)
y2_test.to_csv("data/class_financial_distress_ytest_outliers.csv", index=False)

# Handle Scaling
# techniques1, df_zscore_train1, df_zscore_test1, df_minmax_train1, df_minmax_test1 = data_processing1.handle_scaling()
# print(f"\nForm the plots we conclude that the best approach for the Scaling of the {data_loader1.file_tag} dataset is Standard\n")
# data_processing1.apply_best_scaling_approach('Standard', techniques1,
#                                              df_zscore_train1.drop(columns=[data_loader1.target]), df_zscore_train1[data_loader1.target],
#                                              df_zscore_test1.drop(columns=[data_loader1.target]), df_zscore_test1[data_loader1.target])
#
# techniques2, df_zscore_train2, df_zscore_test2, df_minmax_train2, df_minmax_test2 = data_processing2.handle_scaling()
# print(f"\nForm the plots we conclude that the best approach for the Scaling of the {data_loader2.file_tag} dataset is Standard\n")
# data_processing2.apply_best_scaling_approach('Standard', techniques2,
#                                              df_zscore_train2.drop(columns=[data_loader2.target]), df_zscore_train2[data_loader2.target],
#                                              df_zscore_test2.drop(columns=[data_loader2.target]), df_zscore_test2[data_loader2.target])

# Save the data
# X1_train, X1_test, y1_train, y1_test = (data_processing1.X_train, data_processing1.X_test,
#                                         data_processing1.y_train, data_processing1.y_test)
#
# X2_train, X2_test, y2_train, y2_test = (data_processing2.X_train, data_processing2.X_test,
#                                         data_processing2.y_train, data_processing2.y_test)
#
# X1_train.to_csv("data/class_ny_arrests_Xtrain_scaling.csv", index=False)
# X1_test.to_csv("data/class_ny_arrests_Xtest_scaling.csv", index=False)
# y1_train.to_csv("data/class_ny_arrests_ytrain_scaling.csv", index=False)
# y1_test.to_csv("data/class_ny_arrests_ytest_scaling.csv", index=False)
#
# X2_train.to_csv("data/class_financial_distress_Xtrain_scaling.csv", index=False)
# X2_test.to_csv("data/class_financial_distress_Xtest_scaling.csv", index=False)
# y2_train.to_csv("data/class_financial_distress_ytrain_scaling.csv", index=False)
# y2_test.to_csv("data/class_financial_distress_ytest_scaling.csv", index=False)

# Handle Balancing
# techniques1, df_under_X1, df_under_y1, df_over_X1, df_over_y1, smote_X1, smote_y1 = data_processing1.handle_balancing()
# print(f"\nFrom the plots we conclude that the best approach for the Balancing of the {data_loader1.file_tag} dataset is Oversampling\n")
# data_processing1.apply_best_balancing_approach('Oversampling', techniques1, df_over_X1, df_over_y1)
#
# techniques2, df_under_X2, df_under_y2, df_over_X2, df_over_y2, smote_X2, smote_y2 = data_processing2.handle_balancing()
# print(f"\nFrom the plots we conclude that the best approach for the Balancing of the {data_loader2.file_tag} dataset is SMOTE\n")
# data_processing2.apply_best_balancing_approach('Oversampling', techniques2, df_over_X2, df_over_y2)
#
# # Save the data
# X1_train, X1_test, y1_train, y1_test = (data_processing1.X_train, data_processing1.X_test,
#                                         data_processing1.y_train, data_processing1.y_test)
#
# X2_train, X2_test, y2_train, y2_test = (data_processing2.X_train, data_processing2.X_test,
#                                         data_processing2.y_train, data_processing2.y_test)

# Handle Feature Selection
# data_processing1.handle_feature_selection()
# data_processing2.handle_feature_selection()

# X1_train.to_csv("data/class_ny_arrests_Xtrain_preparation.csv", index=False)
# X1_test.to_csv("data/class_ny_arrests_Xtest_preparation.csv", index=False)
# y1_train.to_csv("data/class_ny_arrests_ytrain_preparation.csv", index=False)
# y1_test.to_csv("data/class_ny_arrests_ytest_preparation.csv", index=False)
#
# X2_train.to_csv("data/class_financial_distress_Xtrain_preparation.csv", index=False)
# X2_test.to_csv("data/class_financial_distress_Xtest_preparation.csv", index=False)
# y2_train.to_csv("data/class_financial_distress_ytrain_preparation.csv", index=False)
# y2_test.to_csv("data/class_financial_distress_ytest_preparation.csv", index=False)

# X1_train = pd.read_csv("data/class_ny_arrests_Xtrain_preparation.csv")
# X1_test = pd.read_csv("data/class_ny_arrests_Xtest_preparation.csv")
# y1_train = pd.read_csv("data/class_ny_arrests_ytrain_preparation.csv")
# y1_test = pd.read_csv("data/class_ny_arrests_ytest_preparation.csv")
#
# X2_train = pd.read_csv("data/class_financial_distress_Xtrain_preparation.csv")
# X2_test = pd.read_csv("data/class_financial_distress_Xtest_preparation.csv")
# y2_train = pd.read_csv("data/class_financial_distress_ytrain_preparation.csv")
# y2_test = pd.read_csv("data/class_financial_distress_ytest_preparation.csv")

# %% 3- Data Modeling

# Data Modeling
data_modeling1 = DataModeling(data_loader1, X1_train, X1_test, y1_train, y1_test)
data_modeling2 = DataModeling(data_loader2, X2_train, X2_test, y2_train, y2_test)

# Naive Bayes
# data_modeling1.naive_bayes()
# data_modeling2.naive_bayes()

# KNN
# data_modeling1.knn()
# data_modeling2.knn()

# Decision Tree
# data_modeling1.decision_tree()
# data_modeling2.decision_tree()

# MLP
# data_modeling1.mlp()
# data_modeling2.mlp()