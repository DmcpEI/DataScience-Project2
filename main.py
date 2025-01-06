# %% Class Imports
import pandas as pd
import json
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

data_processing1.pre_encode_variables_classification()

# Data Dimensionality
# data_profiling1.plot_records_variables_classification()
# data_profiling2.plot_records_variables_classification()
# data_profiling1.plot_variable_types_classification()
# data_profiling2.plot_variable_types_classification()
# data_profiling1.plot_missing_values_classification()
# data_profiling2.plot_missing_values_classification()

# Data Distribution
# data_profiling1.plot_global_boxplots_classification()
# data_profiling2.plot_global_boxplots_classification()
# data_profiling1.plot_single_variable_boxplots_classification()
# data_profiling2.plot_single_variable_boxplots_classification()
# data_profiling1.plot_histograms_classification()
# data_profiling2.plot_histograms_classification()
# data_profiling1.plot_histograms_distribution_classification()
# data_profiling2.plot_histograms_distribution_classification()
# data_profiling1.plot_outlier_comparison_classification()
# data_profiling2.plot_outlier_comparison_classification()
# data_profiling1.plot_class_distribution_classification()
# data_profiling2.plot_class_distribution_classification()

# Data Granularity
# .plot_date_granularity_analysis_classification()
# data_profiling1.plot_location_granularity_analysis_classification()
# data_profiling1.plot_law_code_granularity_analysis_classification()
# data_profiling1.plot_borough_granularity_analysis_classification()
# data_profiling1.plot_age_granularity_analysis_classification()
# data_profiling1.plot_race_granularity_analysis_classification()

# Data Sparsity
# data_profiling1.plot_sparsity_analysis_classification()
# data_profiling2.plot_sparsity_analysis_classification()
# data_profiling1.plot_sparsity_analysis_per_class_classification()
# data_profiling2.plot_sparsity_analysis_per_class_classification()

# Data Encoding of NY Arrests
data_processing1.encode_variables_classification()
# Save the encoded data
data_loader1.data.to_csv("data/class_ny_arrests_encoded.csv", index=True)

# Data Correlation
# data_profiling1.plot_correlation_analysis_classification()
# data_profiling2.plot_correlation_analysis_classification()

# %% 2- Data Processing

# Drop False Predictors
data_processing1.drop_variables_classification()
data_processing2.drop_variables_classification()

# Save the data
data_processing1.data_loader.data.to_csv("data/class_ny_arrests_drop.csv", index=False)
data_processing2.data_loader.data.to_csv("data/class_financial_distress_drop.csv", index=False)
# data_processing1.data_loader.data = pd.read_csv("data/class_ny_arrests_drop.csv")
# data_processing2.data_loader.data = pd.read_csv("data/class_financial_distress_drop.csv")

# Handle Missing Values
techniques1 = data_processing1.handle_missing_values_classification()
print(f"\nFrom the plots we conclude that the best approach for the Missing Values of the {data_loader1.file_tag} dataset is Missing Value Removal\n")
data_processing1.apply_best_missing_value_approach_classification('Remove MV', techniques1)

# techniques2 = data_processing2.handle_missing_values_classification()
print(f"\nThe {data_loader2.file_tag} dataset doesnt have missing values\n")

# Save the data
data_processing1.data_loader.data.to_csv("data/class_ny_arrests_MV.csv", index=False)
data_processing2.data_loader.data.to_csv("data/class_financial_distress_MV.csv", index=False)

# Load the data
# data_processing1.data_loader.data = pd.read_csv("data/class_ny_arrests_MV.csv")
# data_processing2.data_loader.data = pd.read_csv("data/class_financial_distress_MV.csv")

# Data Splitting
X1 = data_processing1.data_loader.data.drop(columns=[data_loader1.target])
y1 = data_processing1.data_loader.data[data_loader1.target]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
data_processing1.X_train, data_processing1.y_train, data_processing1.X_test, data_processing1.y_test = X1_train, y1_train, X1_test, y1_test

X2 = data_processing2.data_loader.data.drop(columns=[data_loader2.target])
y2 = data_processing2.data_loader.data[data_loader2.target]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)
data_processing2.X_train, data_processing2.y_train, data_processing2.X_test, data_processing2.y_test = X2_train, y2_train, X2_test, y2_test

# Save the initial split data
X1_train.to_csv("data/class_ny_arrests_Xtrain_initial.csv", index=False)
X1_test.to_csv("data/class_ny_arrests_Xtest_initial.csv", index=False)
y1_train.to_csv("data/class_ny_arrests_ytrain_initial.csv", index=False)
y1_test.to_csv("data/class_ny_arrests_ytest_initial.csv", index=False)

X2_train.to_csv("data/class_financial_distress_Xtrain_initial.csv", index=False)
X2_test.to_csv("data/class_financial_distress_Xtest_initial.csv", index=False)
y2_train.to_csv("data/class_financial_distress_ytrain_initial.csv", index=False)
y2_test.to_csv("data/class_financial_distress_ytest_initial.csv", index=False)

# Handle Outliers
techniques1, df_train_dropped1, df_train_replaced1, df_train_truncated1 = data_processing1.handle_outliers_classification()
print(f"\nFrom the plots we conclude that the best approach for the Outliers of the {data_loader1.file_tag} dataset is to use Replace\n")
data_processing1.apply_best_outliers_approach_classification('Replace', techniques1,
                                              df_train_truncated1.drop(columns=[data_loader1.target]),
                                              df_train_truncated1[data_loader1.target])

techniques2, df_train_dropped2, df_train_replaced2, df_train_truncated2 = data_processing2.handle_outliers_classification()
print(f"\nFrom the plots we conclude that the best approach for the Outliers of the {data_loader2.file_tag} dataset is to truncate them\n")
data_processing2.apply_best_outliers_approach_classification('Truncate', techniques2,
                                              df_train_truncated2.drop(columns=[data_loader2.target]),
                                              df_train_truncated2[data_loader2.target])

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
techniques1, df_zscore_train1, df_zscore_test1, df_minmax_train1, df_minmax_test1 = data_processing1.handle_scaling_classification()
print(f"\nFrom the plots we conclude that the best approach for the Scaling of the {data_loader1.file_tag} dataset is is to keep the original (no scaling)\n")
data_processing1.apply_best_scaling_approach_classification('Original', techniques1,
                                             df_zscore_train1.drop(columns=[data_loader1.target]), df_zscore_train1[data_loader1.target],
                                             df_zscore_test1.drop(columns=[data_loader1.target]), df_zscore_test1[data_loader1.target])

techniques2, df_zscore_train2, df_zscore_test2, df_minmax_train2, df_minmax_test2 = data_processing2.handle_scaling_classification()
print(f"\nFrom the plots we conclude that the best approach for the Scaling of the {data_loader2.file_tag} dataset is to keep the original (no scaling)\n")
data_processing2.apply_best_scaling_approach_classification('Original', techniques2,
                                             df_zscore_train2.drop(columns=[data_loader2.target]), df_zscore_train2[data_loader2.target],
                                             df_zscore_test2.drop(columns=[data_loader2.target]), df_zscore_test2[data_loader2.target])

# Save the data
X1_train, X1_test, y1_train, y1_test = (data_processing1.X_train, data_processing1.X_test,
                                        data_processing1.y_train, data_processing1.y_test)

X2_train, X2_test, y2_train, y2_test = (data_processing2.X_train, data_processing2.X_test,
                                        data_processing2.y_train, data_processing2.y_test)

X1_train.to_csv("data/class_ny_arrests_Xtrain_scaling.csv", index=True)
X1_test.to_csv("data/class_ny_arrests_Xtest_scaling.csv", index=True)
y1_train.to_csv("data/class_ny_arrests_ytrain_scaling.csv", index=True)
y1_test.to_csv("data/class_ny_arrests_ytest_scaling.csv", index=True)

X2_train.to_csv("data/class_financial_distress_Xtrain_scaling.csv", index=True)
X2_test.to_csv("data/class_financial_distress_Xtest_scaling.csv", index=True)
y2_train.to_csv("data/class_financial_distress_ytrain_scaling.csv", index=True)
y2_test.to_csv("data/class_financial_distress_ytest_scaling.csv", index=True)

# Handle Balancing
techniques1, df_under_X1, df_under_y1, df_over_X1, df_over_y1, smote_X1, smote_y1 = data_processing1.handle_balancing_classification()
print(f"\nFrom the plots we conclude that the best approach for the Balancing of the {data_loader1.file_tag} dataset is SMOTE\n")
data_processing1.apply_best_balancing_approach_classification('Undersampling', techniques1, smote_X1, smote_y1)

techniques2, df_under_X2, df_under_y2, df_over_X2, df_over_y2, smote_X2, smote_y2 = data_processing2.handle_balancing_classification()
print(f"\nFrom the plots we conclude that the best approach for the Balancing of the {data_loader2.file_tag} dataset is Undersampling\n")
data_processing2.apply_best_balancing_approach_classification('Undersampling', techniques2, df_under_X2, df_under_y2)

# Save the data
X1_train, X1_test, y1_train, y1_test = (data_processing1.X_train, data_processing1.X_test,
                                        data_processing1.y_train, data_processing1.y_test)

X2_train, X2_test, y2_train, y2_test = (data_processing2.X_train, data_processing2.X_test,
                                        data_processing2.y_train, data_processing2.y_test)

X1_train.to_csv("data/class_ny_arrests_Xtrain_balancing.csv", index=False)
X1_test.to_csv("data/class_ny_arrests_Xtest_balancing.csv", index=False)
y1_train.to_csv("data/class_ny_arrests_ytrain_balancing.csv", index=False)
y1_test.to_csv("data/class_ny_arrests_ytest_balancing.csv", index=False)

X2_train.to_csv("data/class_financial_distress_Xtrain_balancing.csv", index=False)
X2_test.to_csv("data/class_financial_distress_Xtest_balancing.csv", index=False)
y2_train.to_csv("data/class_financial_distress_ytrain_balancing.csv", index=False)
y2_test.to_csv("data/class_financial_distress_ytest_balancing.csv", index=False)

# Handle Feature Selection
# data_processing1.handle_feature_selection_classification()
vars_to_drop_1 = data_processing1.select_variables_to_drop()
data_processing1.apply_feature_selection(vars_to_drop_1, file_tag="ny_arrests_feature_selection")

# data_processing2.handle_feature_selection_classification()
vars_to_drop_2 = data_processing2.select_variables_to_drop()
data_processing2.apply_feature_selection(vars_to_drop_2, file_tag="financial_distress_feature_selection")

# Inspect columns in the processed datasets
print("Columns in X1_train:", X1_train.columns.tolist())
print("Columns in X1_test:", X1_test.columns.tolist())

print("Columns in X2_train:", X2_train.columns.tolist())
print("Columns in X2_test:", X2_test.columns.tolist())
print("=====================================================================================================")

# Save the data
X1_train, X1_test, y1_train, y1_test = (data_processing1.X_train, data_processing1.X_test,
                                        data_processing1.y_train, data_processing1.y_test)

X2_train, X2_test, y2_train, y2_test = (data_processing2.X_train, data_processing2.X_test,
                                        data_processing2.y_train, data_processing2.y_test)
print("=====================================================================================================")
# Inspect columns in the processed datasets
print("Columns in X1_train:", X1_train.columns.tolist())
print("Columns in X1_test:", X1_test.columns.tolist())

print("Columns in X2_train:", X2_train.columns.tolist())
print("Columns in X2_test:", X2_test.columns.tolist())

X1_train.to_csv("data/class_ny_arrests_Xtrain_preparation.csv", index=False)
X1_test.to_csv("data/class_ny_arrests_Xtest_preparation.csv", index=False)
y1_train.to_csv("data/class_ny_arrests_ytrain_preparation.csv", index=False)
y1_test.to_csv("data/class_ny_arrests_ytest_preparation.csv", index=False)

X2_train.to_csv("data/class_financial_distress_Xtrain_preparation.csv", index=False)
X2_test.to_csv("data/class_financial_distress_Xtest_preparation.csv", index=False)
y2_train.to_csv("data/class_financial_distress_ytrain_preparation.csv", index=False)
y2_test.to_csv("data/class_financial_distress_ytest_preparation.csv", index=False)

# X1_train = pd.read_csv("data/class_ny_arrests_Xtrain_preparation.csv")
# X1_test = pd.read_csv("data/class_ny_arrests_Xtest_preparation.csv")
# y1_train = pd.read_csv("data/class_ny_arrests_ytrain_preparation.csv")
# y1_test = pd.read_csv("data/class_ny_arrests_ytest_preparation.csv")

# X2_train = pd.read_csv("data/class_financial_distress_Xtrain_preparation.csv")
# X2_test = pd.read_csv("data/class_financial_distress_Xtest_preparation.csv")
# y2_train = pd.read_csv("data/class_financial_distress_ytrain_preparation.csv")
# y2_test = pd.read_csv("data/class_financial_distress_ytest_preparation.csv")

# %% 3- Data Modeling

# Data Modeling
data_modeling1 = DataModeling(data_loader1, X1_train, X1_test, y1_train, y1_test)
data_modeling2 = DataModeling(data_loader2, X2_train, X2_test, y2_train, y2_test)

# Initialize a dictionary to store results
results_summary = {
    "NY Arrests": {},  # Dataset 1
    "Financial Distress": {}  # Dataset 2
}

# Naive Bayes
evaluation_results_NB_dataset1 = data_modeling1.naive_bayes_classification()
results_summary["NY Arrests"]["Naive Bayes"] = evaluation_results_NB_dataset1
evaluation_results_NB_dataset2 = data_modeling2.naive_bayes_classification()
results_summary["Financial Distress"]["Naive Bayes"] = evaluation_results_NB_dataset2

print("NY Arrests Naive Bayes")
print(evaluation_results_NB_dataset1)
print("Financial Distress Naive Bayes")
print(evaluation_results_NB_dataset2)

# KNN
evaluation_results_KNN_dataset1 = data_modeling1.knn_classification()
results_summary["NY Arrests"]["KNN"] = evaluation_results_KNN_dataset1
evaluation_results_KNN_dataset2 = data_modeling2.knn_classification()
results_summary["Financial Distress"]["KNN"] = evaluation_results_KNN_dataset2

print("NY Arrests KNN")
print(evaluation_results_KNN_dataset1)
print("Financial Distress KNN")
print(evaluation_results_KNN_dataset2)

# Decision Tree
evaluation_results_DT_dataset1 = data_modeling1.decision_tree_classification()
results_summary["NY Arrests"]["Decision Tree"] = evaluation_results_DT_dataset1
evaluation_results_DT_dataset2 = data_modeling2.decision_tree_classification()
results_summary["Financial Distress"]["Decision Tree"] = evaluation_results_DT_dataset2

print("NY Arrests Decision Tree")
print(evaluation_results_DT_dataset1)
print("Financial Distress Decision Tree")
print(evaluation_results_DT_dataset2)

# MLP
evaluation_results_MLP_dataset1 = data_modeling1.mlp_classification()
results_summary["NY Arrests"]["MLP"] = evaluation_results_MLP_dataset1
evaluation_results_MLP_dataset2 = data_modeling2.mlp_classification()
results_summary["Financial Distress"]["MLP"] = evaluation_results_MLP_dataset2

print("NY Arrests MLP")
print(evaluation_results_MLP_dataset1)
print("Financial Distress MLP")
print(evaluation_results_MLP_dataset2)

# Random Forests
evaluation_results_RF_dataset1 = data_modeling1.random_forest_classification()
results_summary["NY Arrests"]["Random Forest"] = evaluation_results_RF_dataset1
evaluation_results_RF_dataset2 = data_modeling2.random_forest_classification()
results_summary["Financial Distress"]["Random Forest"] = evaluation_results_RF_dataset2

print("NY Arrests Random Forest")
print(evaluation_results_RF_dataset1)
print("Financial Distress Random Forest")
print(evaluation_results_RF_dataset2)

# Gradient Boosting
evaluation_results_GB_dataset1 = data_modeling1.gradient_boosting_classification()
results_summary["NY Arrests"]["Gradient Boosting"] = evaluation_results_GB_dataset1
evaluation_results_GB_dataset2 = data_modeling2.gradient_boosting_classification()
results_summary["Financial Distress"]["Gradient Boosting"] = evaluation_results_GB_dataset2

print("NY Arrests Gradient Boosting")
print(evaluation_results_GB_dataset1)
print("Financial Distress Gradient Boosting")
print(evaluation_results_GB_dataset2)

# Save results to a file (optional)
with open("classification_evaluation_results_summary.json", "w") as file:
    json.dump(results_summary, file, indent=4)

# Print the full summary at the end
print("\nFull Evaluation Results Summary:")
print(json.dumps(results_summary, indent=4))