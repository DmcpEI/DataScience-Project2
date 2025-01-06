from math import ceil, log10

import numpy as np
from numpy import arange
import pandas as pd
from pandas import read_csv, DataFrame, Series, Index
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots, show, savefig, figure
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from imblearn.over_sampling import SMOTE

from dslabs_functions import determine_outlier_thresholds_for_var, run_NB, run_KNN, CLASS_EVAL_METRICS, \
    plot_multibar_chart, get_variable_types, concat, plot_multiline_chart, HEIGHT, plot_forecasting_eval, \
    series_train_test_split, plot_forecasting_series, plot_line_chart, ts_aggregation_by, \
    dataframe_temporal_train_test_split


class DataProcessing:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.target = self.data_loader.target
        self.previous_accuracy = {}
        self.X_train = DataFrame()
        self.X_test = DataFrame()
        self.y_train = DataFrame()
        self.y_test = DataFrame()

    def pre_encode_variables_classification(self):

        # Encode AGE_GROUP
        print("Encoding AGE_GROUP...")
        valid_age_groups = {'<18', '18-24', '25-44', '45-64', '65+'}
        self.data_loader.data['AGE_GROUP'] = self.data_loader.data['AGE_GROUP'].where(
            self.data_loader.data['AGE_GROUP'].isin(valid_age_groups), 'UNKNOWN')
        age_group_mapping = {'UNKNOWN': None, '<18': 1, '18-24': 2, '25-44': 3, '45-64': 4, '65+': 5}
        self.data_loader.data['AGE_GROUP'] = self.data_loader.data['AGE_GROUP'].map(age_group_mapping)

        # Encode JURISDICTION_CODE
        self.data_loader.data['JURISDICTION_CODE'] = self.data_loader.data['JURISDICTION_CODE'].apply(lambda x: 'NY' if x < 3 else 'nonNY')

    def encode_law_code(self):
        # Replace NaN with 'UNKNOWN'
        self.data_loader.data['LAW_CODE'] = self.data_loader.data['LAW_CODE'].fillna('UNKNOWN')

        # Extract prefix and suffix, leaving 'UNKNOWN' unchanged
        self.data_loader.data['prefix'] = self.data_loader.data['LAW_CODE'].apply(lambda x: x[:3] if x != 'UNKNOWN' else 'UNK')
        self.data_loader.data['suffix'] = self.data_loader.data['LAW_CODE'].apply(lambda x: x[3:].strip() if x != 'UNKNOWN' else 'UNKNOWN')

        # Initialize LabelEncoders
        prefix_encoder = LabelEncoder()
        suffix_encoder = LabelEncoder()

        # Fit encoders only on unique, non-'UNKNOWN' values
        prefix_values = self.data_loader.data.loc[self.data_loader.data['prefix'] != 'UNK', 'prefix'].unique()
        suffix_values = self.data_loader.data.loc[self.data_loader.data['suffix'] != 'UNKNOWN', 'suffix'].unique()

        prefix_encoder.fit(prefix_values)
        suffix_encoder.fit(suffix_values)

        # Encode prefix and suffix, keeping 'UNKNOWN' as-is
        self.data_loader.data['prefix_encoded'] = self.data_loader.data['prefix'].apply(lambda x: prefix_encoder.transform([x])[0] if x != 'UNK' else None)
        self.data_loader.data['suffix_encoded'] = self.data_loader.data['suffix'].apply(lambda x: suffix_encoder.transform([x])[0] if x != 'UNKNOWN' else None)

        # Calculate number of unique suffixes and multiplier
        num_suffixes = len(suffix_values)
        num_digits = ceil(log10(num_suffixes)) if num_suffixes > 0 else 1
        multiplier = 10 ** num_digits

        # Combine prefix and suffix encoding, handling 'UNKNOWN'
        self.data_loader.data['combined_encoded'] = self.data_loader.data.apply(
            lambda row: int(row['prefix_encoded'] * multiplier + row['suffix_encoded'])  # Force int type
            if pd.notna(row['prefix_encoded']) and pd.notna(row['suffix_encoded'])
            else 'UNKNOWN',
            axis=1
        )

        # Explicitly cast numeric values to `int`, leave 'UNKNOWN' as string
        self.data_loader.data['combined_encoded'] = self.data_loader.data['combined_encoded'].apply(
            lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else x
        )

        # Replace the original LAW_CODE column with combined_encoded
        self.data_loader.data['LAW_CODE'] = self.data_loader.data['combined_encoded']

        # Drop unnecessary columns
        self.data_loader.data = self.data_loader.data.drop(columns=['prefix', 'suffix', 'prefix_encoded', 'suffix_encoded', 'combined_encoded'])

    def encode_variables_classification(self):
        """
        Encodes all symbolic variables in the dataset based on the granularity analysis.
        """

        # Encode ARREST_DATE
        print("Encoding ARREST_DATE...")
        arrest_date = pd.to_datetime(self.data_loader.data['ARREST_DATE'], format='%m/%d/%Y', errors='coerce')
        self.data_loader.data['ARREST_DAY'] = arrest_date.dt.day
        self.data_loader.data['ARREST_MONTH'] = arrest_date.dt.month
        self.data_loader.data['ARREST_QUARTER'] = arrest_date.dt.quarter
        self.data_loader.data['ARREST_YEAR'] = arrest_date.dt.year
        self.data_loader.data['ARREST_DAYOFWEEK'] = arrest_date.dt.dayofweek
        self.data_loader.data.drop(columns=['ARREST_DATE'], inplace=True)

        # Cyclic Encoding for ARREST_QUARTER
        print("Applying cyclic encoding to ARREST_QUARTER...")
        self.data_loader.data['ARREST_QUARTER_SIN'] = np.sin(2 * np.pi * self.data_loader.data['ARREST_QUARTER'] / 4)
        self.data_loader.data['ARREST_QUARTER_COS'] = np.cos(2 * np.pi * self.data_loader.data['ARREST_QUARTER'] / 4)
        self.data_loader.data.drop(columns=['ARREST_QUARTER'], inplace=True)  # Drop original column

        # Cyclic Encoding for ARREST_DAYOFWEEK
        print("Applying cyclic encoding to ARREST_DAYOFWEEK...")
        self.data_loader.data['ARREST_DAYOFWEEK_SIN'] = np.sin(
            2 * np.pi * self.data_loader.data['ARREST_DAYOFWEEK'] / 7)
        self.data_loader.data['ARREST_DAYOFWEEK_COS'] = np.cos(
            2 * np.pi * self.data_loader.data['ARREST_DAYOFWEEK'] / 7)
        self.data_loader.data.drop(columns=['ARREST_DAYOFWEEK'], inplace=True)  # Drop original column

        # Encode PD_DESC
        print("Encoding PD_DESC...")

        vectorizer_pd = CountVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]+\b', max_features=20)
        word_matrix_pd = vectorizer_pd.fit_transform(self.data_loader.data['PD_DESC'].fillna(''))

        top_words_pd = vectorizer_pd.get_feature_names_out()
        print(f"Top Words in PD_DESC: {top_words_pd}")

        for word in top_words_pd:
            self.data_loader.data[f'PD_HAS_{word.upper()}'] = (
                self.data_loader.data['PD_DESC'].str.contains(word, case=False, na=False).astype(int))

        self.data_loader.data.drop(columns=['PD_DESC'], inplace=True)

        # Encode OFNS_DESC
        print("Encoding OFNS_DESC...")
        vectorizer_ofns = CountVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]+\b', max_features=20)
        word_matrix_ofns = vectorizer_ofns.fit_transform(self.data_loader.data['OFNS_DESC'].fillna(''))

        top_words_ofns = vectorizer_ofns.get_feature_names_out()
        print(f"Top Words in OFNS_DESC: {top_words_ofns}")

        for word in top_words_ofns:
            self.data_loader.data[f'OFNS_HAS_{word.upper()}'] = (
                self.data_loader.data['OFNS_DESC'].str.contains(word, case=False, na=False).astype(int))

        self.data_loader.data.drop(columns=['OFNS_DESC'], inplace=True)

        # # Encode PD_DESC
        # print("Encoding PD_DESC...")
        # label_encoder = LabelEncoder()
        # self.data_loader.data['PD_DESC'] = label_encoder.fit_transform(
        #     self.data_loader.data['PD_DESC'].fillna('UNKNOWN'))
        #
        # # Encode OFNS_DESC
        # print("Encoding OFNS_DESC...")
        # label_encoder = LabelEncoder()
        # self.data_loader.data['OFNS_DESC'] = label_encoder.fit_transform(
        #     self.data_loader.data['OFNS_DESC'].fillna('UNKNOWN'))

        # Encode LAW_CODE
        print("Encoding LAW_CODE...")
        self.encode_law_code()

        # Encode LAW_CAT_CD, ARREST_BORO, PERP_SEX, PERP_RACE
        print("Encoding other variables...")
        mapping = {
            'JURISDICTION_CODE': {'NY': 0, 'nonNY': 1},
            'LAW_CAT_CD': {'F': 1, 'M': 0},
            'ARREST_BORO': {'M': 1, 'B': 2, 'Q': 3, 'K': 4, 'S': 5},
            'PERP_SEX': {'M': 1, 'F': 0},
            'PERP_RACE': {
                'UNKNOWN': None, 'BLACK': 1, 'BLACK HISPANIC': 2, 'ASIAN / PACIFIC ISLANDER': 3,
                'AMERICAN INDIAN/ALASKAN NATIVE': 4, 'WHITE HISPANIC': 5,
                'WHITE': 6, 'OTHER': 7
            }
        }

        for col, mapping_dict in mapping.items():
            self.data_loader.data[col] = self.data_loader.data[col].map(mapping_dict)

        print("All symbolic variables encoded.")

    def drop_variables_classification(self):
        """
        Drop variables that are false predictions or irrelevant
        """
        if self.target == "JURISDICTION_CODE":

            self.data_loader.data.drop(columns=['ARREST_KEY'], inplace=True)
            print("\nDropped 'ARREST_KEY' variable for being irrelevant for the classification task.")

        elif self.target == "CLASS":

            self.data_loader.data.drop(columns=['Financial Distress'], inplace=True)
            print("\nDropped 'Financial Distress' variable for being a false predictor.")

    def evaluate_step_classification(self, X_train, X_test, y_train, y_test, dataset, file_tag, metric="accuracy", plot_title="Model Evaluation", is_scalling = False):
        """
        Evaluates the model's performance on a given dataset and generates a comparison plot.

        Parameters:
        - X (pd.DataFrame): The dataset to evaluate.
        - y (pd.Series): The target variable.
        - test_size (float): The proportion of the dataset to include in the test split.
        - metric (str): The primary evaluation metric for model selection.
        - plot_title (str): Title for the evaluation plot.
        - file_tag (str): File tag for saving the evaluation plot.

        Returns:
        - eval_results (dict): A dictionary containing model performance metrics.
        """

        # Dictionary to store evaluation results
        eval_results = {}

        if is_scalling == False:

            # Run Naive Bayes and KNN models and evaluate them
            eval_NB = run_NB(X_train, y_train, X_test, y_test, metric=metric)
            eval_KNN = run_KNN(X_train, y_train, X_test, y_test, metric=metric)

            # Combine the results for plotting
            for metric_name in CLASS_EVAL_METRICS:
                eval_results[metric_name] = [eval_NB[metric_name], eval_KNN[metric_name]]

            # Plot the results as a multibar chart
            figure()
            plot_multibar_chart(["NB", "KNN"], eval_results, title=plot_title, percentage=True)
            savefig(f"graphs/classification/data_preparation/{dataset}_{file_tag}_eval.png")
            show()

            return eval_results

        else:

            # Run KNN models and evaluate them
            eval_KNN = run_KNN(X_train, y_train, X_test, y_test, metric=metric)

            # Combine the results for plotting
            for metric_name in CLASS_EVAL_METRICS:
                eval_results[metric_name] = [eval_KNN[metric_name]]

            # Plot the results as a multibar chart
            figure()
            plot_multibar_chart(["KNN"], eval_results, title=plot_title, percentage=True)
            savefig(f"graphs/classification/data_preparation/{dataset}_{file_tag}_eval.png")
            show()

            return eval_results

    def handle_missing_values_classification(self):
        """
        Handles missing values using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Mean & Most Frequent Imputation
        - Median & Most Frequent Imputation
        - Row Removal (drops rows with any missing values)
        """

        print(f"\n\nHandling missing values for the {self.data_loader.file_tag} dataset...")

        # Data setup
        X = self.data_loader.data.drop(columns=[self.target])
        y = self.data_loader.data[self.target]

        # Use get_variable_types to categorize variables
        variable_types = get_variable_types(X)
        numeric_columns = variable_types["numeric"]
        symbolic_columns = variable_types["symbolic"]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Dictionary to store results
        techniques = {}

        # Row Removal on training data
        X_train_removed = X_train.dropna()
        y_train_removed = y_train[X_train_removed.index]

        if not X_train_removed.empty:  # Ensure removal doesn't leave the dataset empty
            # Ensure the test set remains consistent
            X_test_removed = X_test.dropna()
            y_test_removed = y_test[X_test_removed.index]

            techniques['Remove MV'] = self.evaluate_step_classification(
                X_train_removed, X_test_removed, y_train_removed, y_test_removed,
                self.data_loader.file_tag, "MV_Row_Removal",
                plot_title=f"Evaluation for {self.data_loader.file_tag} - MV Row Removal"
            )

        else:
            print("\nRow Removal skipped (would result in an empty dataset).")
            techniques['Remove MV'] = {'knn': 0, 'nb': 0, 'average_accuracy': 0}

        # Mean & Most Frequent Imputation
        imputer_mean_numeric = SimpleImputer(strategy='mean')
        imputer_most_frequent = SimpleImputer(strategy='most_frequent')

        X_train_mean = X_train.copy()
        X_test_mean = X_test.copy()
        if numeric_columns:
            X_train_mean[numeric_columns] = imputer_mean_numeric.fit_transform(X_train[numeric_columns])
            X_test_mean[numeric_columns] = imputer_mean_numeric.transform(X_test[numeric_columns])
        if symbolic_columns:
            X_train_mean[symbolic_columns] = imputer_most_frequent.fit_transform(X_train[symbolic_columns])
            X_test_mean[symbolic_columns] = imputer_most_frequent.transform(X_test[symbolic_columns])

        techniques['Mean & Most Frequent'] = self.evaluate_step_classification(
            X_train_mean, X_test_mean, y_train, y_test,
            self.data_loader.file_tag, "MV_Mean_Most_Frequent_Imputation",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - MV Mean & Most Frequent Imputation"
        )

        # Median & Most Frequent Imputation
        imputer_median_numeric = SimpleImputer(strategy='median')

        X_train_median = X_train.copy()
        X_test_median = X_test.copy()
        if numeric_columns:
            X_train_median[numeric_columns] = imputer_median_numeric.fit_transform(X_train[numeric_columns])
            X_test_median[numeric_columns] = imputer_median_numeric.transform(X_test[numeric_columns])
        if symbolic_columns:
            X_train_median[symbolic_columns] = imputer_most_frequent.fit_transform(X_train[symbolic_columns])
            X_test_median[symbolic_columns] = imputer_most_frequent.transform(X_test[symbolic_columns])

        techniques['Median & Most Frequent'] = self.evaluate_step_classification(
            X_train_median, X_test_median, y_train, y_test,
            self.data_loader.file_tag, "MV_Median_Most_Frequent_Imputation",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - MV Median & Most Frequent Imputation"
        )

        # Print all performances at the end
        print("\nSummary of Missing Value Handling Techniques:")
        for technique, performance in techniques.items():
            print(f"{technique} performance: {performance}")

        return techniques

    def apply_best_missing_value_approach_classification(self, best_technique, techniques):
        """
        Applies the best missing value handling technique to the dataset.
        """
        if best_technique == 'Remove MV':
            print("Applying Row Removal...")
            self.data_loader.data = self.data_loader.data.dropna()

        elif best_technique == 'Mean & Most Frequent':
            print("Applying Mean & Most Frequent Imputation...")
            imputer_mean_numeric = SimpleImputer(strategy='mean')
            imputer_most_frequent = SimpleImputer(strategy='most_frequent')

            numeric_columns = get_variable_types(self.data_loader.data.drop(columns=[self.target]))["numeric"]
            symbolic_columns = get_variable_types(self.data_loader.data.drop(columns=[self.target]))["symbolic"]

            if numeric_columns:
                self.data_loader.data[numeric_columns] = imputer_mean_numeric.fit_transform(
                    self.data_loader.data[numeric_columns])
            if symbolic_columns:
                self.data_loader.data[symbolic_columns] = imputer_most_frequent.fit_transform(
                    self.data_loader.data[symbolic_columns])

        elif best_technique == 'Median & Most Frequent':
            print("Applying Median & Most Frequent Imputation...")
            imputer_median_numeric = SimpleImputer(strategy='median')
            imputer_most_frequent = SimpleImputer(strategy='most_frequent')

            numeric_columns = get_variable_types(self.data_loader.data.drop(columns=[self.target]))["numeric"]
            symbolic_columns = get_variable_types(self.data_loader.data.drop(columns=[self.target]))["symbolic"]

            if numeric_columns:
                self.data_loader.data[numeric_columns] = imputer_median_numeric.fit_transform(
                    self.data_loader.data[numeric_columns])
            if symbolic_columns:
                self.data_loader.data[symbolic_columns] = imputer_most_frequent.fit_transform(
                    self.data_loader.data[symbolic_columns])

        self.previous_accuracy = techniques[best_technique]

    def handle_outliers_classification(self):
        """
        Handles outliers using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Original Dataset (no outlier removal)
        - Replacing Outliers
        - Truncating Outliers
        """

        print(f"\n\nHandling outliers for the {self.data_loader.file_tag} dataset...")

        data_train = pd.concat([self.X_train, self.y_train], axis=1)

        # Dictionary to store results for different techniques
        techniques = {}

        # Original Dataset (Baseline)
        print("Original train data shape:", data_train.shape)
        techniques['Original'] = self.previous_accuracy

        numeric_vars_train = self.X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_vars_train:
            print("There are no numeric variables to process for outliers.")
            return

        # Drop Outliers
        df_train_dropped = data_train.copy()
        summary = df_train_dropped[numeric_vars_train].describe()

        for var in numeric_vars_train:
            top, bottom = determine_outlier_thresholds_for_var(summary[var], threshold=5)
            outliers = df_train_dropped[(df_train_dropped[var] > top) | (df_train_dropped[var] < bottom)]
            df_train_dropped.drop(outliers.index, axis=0, inplace=True)

        results_drop = self.evaluate_step_classification(df_train_dropped.drop(columns=[self.target]), self.X_test,
                                          df_train_dropped[self.target], self.y_test, self.data_loader.file_tag,
                                          "Outlier_Drop", plot_title=f"Evaluation for {self.data_loader.file_tag} - Outlier Drop")
        print("Train data shape after dropping outliers:", df_train_dropped.shape)
        techniques['Drop'] = results_drop

        # Replace Outliers with Median
        df_train_replaced = data_train.copy()

        for var in numeric_vars_train:
            top, bottom = determine_outlier_thresholds_for_var(summary[var], threshold=5)
            median = df_train_replaced[var].median()
            df_train_replaced[var] = df_train_replaced[var].apply(lambda x: median if x > top or x < bottom else x)

        results_replace = self.evaluate_step_classification(df_train_replaced.drop(columns=[self.target]), self.X_test,
                                             df_train_replaced[self.target], self.y_test, self.data_loader.file_tag,
                                             "Outlier_Replace", plot_title=f"Evaluation for {self.data_loader.file_tag} - Outlier Replace")
        print("Train data shape after replacing outliers:", df_train_replaced.shape)
        print("Train data shape description after replacing outliers:\n", df_train_replaced.describe())
        techniques['Replace'] = results_replace

        # Truncate Outliers
        df_train_truncated = data_train.copy()

        for var in numeric_vars_train:
            top, bottom = determine_outlier_thresholds_for_var(summary[var], threshold=5)
            df_train_truncated[var] = df_train_truncated[var].apply(lambda x: top if x > top else (bottom if x < bottom else x))

        results_truncate = self.evaluate_step_classification(df_train_truncated.drop(columns=[self.target]), self.X_test,
                                              df_train_truncated[self.target], self.y_test, self.data_loader.file_tag,
                                              "Outlier_Truncate", plot_title=f"Evaluation for {self.data_loader.file_tag} - Outlier Truncate")
        print("Train data shape after replacing outliers:", df_train_truncated.shape)
        print("Train data shape description after replacing outliers:\n", df_train_truncated.describe())
        techniques['Truncate'] = results_truncate

        # Print all performances at the end
        print("\nSummary of Outlier Handling Techniques:")
        for technique, performance in techniques.items():
            print(f"{technique} performance: {performance}")

        return techniques, df_train_dropped, df_train_replaced, df_train_truncated

    def apply_best_outliers_approach_classification(self, best_technique, techniques, X_train = None, y_train = None):

        numeric_vars = self.X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_vars:
            print("There are no numeric variables to process for outliers.")
            return
        else:
            if best_technique == 'Original':
                print("No changes made to the dataset (original data retained).")
            else:
                print(f"Applying {best_technique} Outliers...")
                self.X_train = X_train
                self.y_train = y_train

        self.previous_accuracy = techniques[best_technique]

    def handle_scaling_classification(self):
        """
        Handles scaling using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Standard Scaler
        - MinMax Scaler
        """

        print(f"\n\nHandling scaling for the {self.data_loader.file_tag} dataset...")

        # Combine data for easier handling
        data_train = pd.concat([self.X_train, self.y_train], axis=1)
        data_test = pd.concat([self.X_test, self.y_test], axis=1)

        # List to store results for different techniques
        techniques = {}

        # Original Dataset (Baseline)
        techniques['Original'] = self.previous_accuracy

        # Scaling with Standard Scaler

        # Separate features and target
        target_data_train: Series = data_train.pop(self.target)
        target_data_test: Series = data_test.pop(self.target)

        # Fit the scaler on training data
        transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True)
        transf.fit(data_train)

        # Transform both training and test sets using the same scaler
        df_zscore_train = pd.DataFrame(transf.transform(data_train), index=data_train.index, columns=data_train.columns)
        df_zscore_test = pd.DataFrame(transf.transform(data_test), index=data_test.index, columns=data_test.columns)

        # Add the target column back
        df_zscore_train[self.target] = target_data_train
        df_zscore_test[self.target] = target_data_test

        # Evaluate the performance of the Standard Scaler
        results_standard = self.evaluate_step_classification(
            df_zscore_train.drop(columns=[self.target]),
            df_zscore_test.drop(columns=[self.target]),
            df_zscore_train[self.target],
            df_zscore_test[self.target],
            self.data_loader.file_tag, "Scaling_Standard",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - Standard Scaling",
            is_scalling = True
        )
        techniques['Standard'] = results_standard

        # Scaling with MinMax Scaler

        # Fit the MinMaxScaler on training data
        transf_minmax: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        transf_minmax.fit(data_train)

        # Transform both training and test sets using the MinMaxScaler
        df_minmax_train = pd.DataFrame(transf_minmax.transform(data_train), index=data_train.index,
                                       columns=data_train.columns)
        df_minmax_test = pd.DataFrame(transf_minmax.transform(data_test), index=data_test.index,
                                      columns=data_test.columns)

        # Add the target column back
        df_minmax_train[self.target] = target_data_train
        df_minmax_test[self.target] = target_data_test

        # Evaluate the performance of the MinMax Scaler
        results_minmax = self.evaluate_step_classification(
            df_minmax_train.drop(columns=[self.target]),
            df_minmax_test.drop(columns=[self.target]),
            df_minmax_train[self.target],
            df_minmax_test[self.target],
            self.data_loader.file_tag, "Scaling_MinMax",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - MinMax Scaling",
            is_scalling = True
        )
        techniques['MinMax'] = results_minmax

        # Print all performances at the end
        print("\nSummary of Scaling Techniques:")
        for technique, performance in techniques.items():
            print(f"{technique} performance: {performance}")

        return techniques, df_zscore_train, df_zscore_test, df_minmax_train, df_minmax_test

    def apply_best_scaling_approach_classification(self, best_technique, techniques, X_train = None, y_train = None, X_test = None, y_test = None):
        """
        Applies the best scaling technique to the dataset.
        """
        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        else:
            print(f"Applying {best_technique} Scaler...")
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

        self.previous_accuracy = techniques[best_technique]

    def handle_balancing_classification(self):
        """
        Handles class imbalance using undersampling, oversampling, and SMOTE techniques,
        and evaluates the performance of each.
        """
        print(f"\n\nHandling balancing for the {self.data_loader.file_tag} dataset...\n")

        # Get class counts and identify minority and majority classes
        target_count = self.y_train.value_counts()
        positive_class = target_count.idxmin()
        negative_class = target_count.idxmax()

        print(f"Class distribution before balancing: {target_count.to_dict()}")

        techniques = {"Original": self.previous_accuracy}

        # Random Undersampling
        positives = self.X_train[self.y_train == positive_class]
        negatives = self.X_train[self.y_train == negative_class].sample(len(positives), random_state=42)
        X_train_under = pd.concat([positives, negatives])
        y_train_under = pd.concat([self.y_train[positives.index], self.y_train[negatives.index]])
        techniques["Undersampling"] = self.evaluate_step_classification(
            X_train_under, self.X_test, y_train_under, self.y_test, self.data_loader.file_tag,
            "Balancing_Undersampling",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - Undersampling"
        )

        # Random Oversampling
        negatives = self.X_train[self.y_train == negative_class]
        positives = self.X_train[self.y_train == positive_class].sample(len(negatives), replace=True, random_state=42)
        X_train_over = pd.concat([positives, negatives])
        y_train_over = pd.concat([self.y_train[positives.index], self.y_train[negatives.index]])
        techniques["Oversampling"] = self.evaluate_step_classification(
            X_train_over, self.X_test, y_train_over, self.y_test, self.data_loader.file_tag,
            "Balancing_Oversampling",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - Oversampling"
        )

        # SMOTE
        smote = SMOTE(sampling_strategy="minority", random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        techniques["SMOTE"] = self.evaluate_step_classification(
            X_train_smote, self.X_test, y_train_smote, self.y_test, self.data_loader.file_tag,
            "Balancing_SMOTE",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - SMOTE"
        )

        # Print all performances at the end
        print("\nSummary of Balancing Techniques:")
        for technique, performance in techniques.items():
            print(f"{technique} performance: {performance}")

        return techniques, X_train_under, y_train_under, X_train_over, y_train_over, X_train_smote, y_train_smote

    def apply_best_balancing_approach_classification(self, best_technique, techniques, X_train = None, y_train = None):
        """
        Applies the best balancing technique to the dataset.
        """
        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        else:
            print(f"Applying {best_technique}...")
            self.X_train = X_train
            self.y_train = y_train

        self.previous_accuracy = techniques[best_technique]

    def handle_feature_selection_classification(self):

        print(f"\n\nHandling feature selection for the {self.data_loader.file_tag} dataset...\n")

        # vars_to_drop = self.select_low_variance_variables(max_threshold=3)
        # print(f"Variables to drop: {vars_to_drop}")
        self.study_variance(self.X_train, self.X_test, self.y_train, self.y_test,
                            max_threshold=1, lag=0.2, metric="f1", file_tag=self.data_loader.file_tag)


        vars_to_drop_2 = self.select_redundant_variables(min_threshold=0.9)
        print(f"Variables to drop: {vars_to_drop_2}")
        self.study_redundancy_for_feature_selection(self.X_train, self.X_test, self.y_train, self.y_test,
                                                    min_threshold=0.3, lag=0.3, metric="f1",
                                                    file_tag=self.data_loader.file_tag)



        self.apply_feature_selection(vars_to_drop_2, file_tag="ny_arrests_feature_selection")

    def select_low_variance_variables(self, max_threshold: float) -> list[str]:
        """
        Identifies low-variance variables to drop based on the threshold.
        :param max_threshold: Maximum variance threshold for feature selection.
        :return: List of variable names to drop.
        """
        data: DataFrame = self.data_loader.data
        summary5: DataFrame = data.describe()
        vars2drop: Index[str] = summary5.columns[
            summary5.loc["std"] * summary5.loc["std"] < max_threshold
            ]
        vars2drop = vars2drop.drop(self.target) if self.target in vars2drop else vars2drop
        return list(vars2drop.values)

    def study_variance(self, X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                       max_threshold: float = 1, lag: float = 0.1,
                       metric: str = "accuracy", file_tag: str = ""):
        """
        Studies the impact of low variance thresholds on model performance.

        :param max_threshold: Maximum variance threshold to test.
        :param lag: Step size for thresholds.
        :param metric: Evaluation metric (e.g., accuracy, recall).
        :param file_tag: Tag to use for saving the plot file.
        """

        # Generate a list of threshold options to test the impact of different thresholds
        threshold_options: list[float] = [
            round(i * lag, 3) for i in range(1, ceil(max_threshold / lag + lag))
        ]
        results: dict[str, dict[float, list]] = {"NB": {}, "KNN": {}}  # Dictionary to store results
        summary: DataFrame = X_train.describe()  # Get summary statistics of the training data

        # Iterate over each threshold option
        for thresh in threshold_options:
            # Identify variables with variance below the threshold
            vars2drop: Index[str] = summary.columns[
                summary.loc["std"] * summary.loc["std"] < thresh
                ]
            # if the target is in the variables to drop, remove it from that list
            if self.target in vars2drop:
                vars2drop = vars2drop.drop(self.target)

            # Drop the low variance variables from the training and testing datasets
            X_train_processed: DataFrame = X_train.drop(vars2drop, axis=1, inplace=False)
            X_test_processed: DataFrame = X_test.drop(vars2drop, axis=1, inplace=False)

            # Evaluate the model performance with the selected variables
            eval: dict[str, list] | None = self.evaluate_step_classification(
                X_train_processed, X_test_processed, y_train, y_test,
                dataset=self.data_loader.file_tag, file_tag=f"{file_tag}_low_var_{thresh}",
                metric=metric, plot_title=f"Evaluation for {self.data_loader.file_tag} - Low Variance ({thresh})"
            )
            if eval:
                print("Evaluation results for threshold", thresh, ":", eval)
                results["NB"][thresh] = eval[metric][0]
                results["KNN"][thresh] = eval[metric][1]

            #print(results)

        # Prepare lists for each model
        nb_metrics = []
        knn_metrics = []
        for threshold in threshold_options:
            # For each threshold, get the corresponding metric value for Naive Bayes and KNN
            nb_metrics.append(results["NB"].get(threshold, None))  # Directly append the value if it's a float
            knn_metrics.append(results["KNN"].get(threshold, None))  # Same for KNN

        # Plot the results
        figure(figsize=(2 * HEIGHT, HEIGHT))
        plot_multiline_chart(
            threshold_options,
            {"NB": nb_metrics, "KNN": knn_metrics},
            title=f"{file_tag} variance study ({metric})",
            xlabel="Variance Threshold",
            ylabel=metric,
            percentage=True,
        )
        savefig(f"graphs/classification/data_preparation/{file_tag}_fs_low_var_{metric}_study.png")
        show()

    def apply_feature_selection(self, vars2drop: list[str], file_tag: str):
        """
        Applies feature selection by dropping specified variables from training and testing datasets.

        :param vars2drop: List of variables to drop.
        :param file_tag: Tag to include in the filenames of the saved datasets.
        """
        # Ensure vars2drop is valid
        if not isinstance(vars2drop, list) or len(vars2drop) == 0:
            print("No variables to drop.")
            return

        # Ensure vars2drop columns exist in both X_train and X_test
        vars_to_remove = [var for var in vars2drop if var in self.X_train.columns]

        # Drop selected variables from X_train and X_test
        self.X_train = self.X_train.drop(columns=vars_to_remove, inplace=False)
        self.X_test = self.X_test.drop(columns=vars_to_remove, inplace=False)

        # Save processed datasets
        self.X_train.to_csv(f"data/{file_tag}_train.csv", index=False)
        self.X_test.to_csv(f"data/{file_tag}_test.csv", index=False)

        # Print summary
        print(f"Feature selection applied. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        print(f"Variables dropped: {vars_to_remove}")

    def select_redundant_variables(self, min_threshold) -> list:
        df: DataFrame = self.data_loader.data.drop(self.target, axis=1, inplace=False)
        corr_matrix: DataFrame = abs(df.corr())
        variables: Index[str] = corr_matrix.columns
        vars2drop: list = []
        for v1 in variables:
            vars_corr: Series = (corr_matrix[v1]).loc[corr_matrix[v1] >= min_threshold]
            vars_corr.drop(v1, inplace=True)
            if len(vars_corr) > 1:
                lst_corr = list(vars_corr.index)
                for v2 in lst_corr:
                    if v2 not in vars2drop:
                        vars2drop.append(v2)
        return vars2drop

    def study_redundancy_for_feature_selection(
            self, X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
            min_threshold: float = 0.90,
            lag: float = 0.05,
            metric: str = "accuracy",
            file_tag: str = "",
    ) -> dict:
        options: list[float] = [
            round(min_threshold + i * lag, 3)
            for i in range(ceil((1 - min_threshold) / lag) + 1)
        ]

        df: DataFrame = X_train
        corr_matrix: DataFrame = abs(df.corr())
        variables: Index[str] = corr_matrix.columns
        results: dict[str, dict[float, list]] = {"NB": {}, "KNN": {}}  # Dictionary to store results
        for thresh in options:
            vars2drop: list = []
            for v1 in variables:
                vars_corr: Series = (corr_matrix[v1]).loc[corr_matrix[v1] >= thresh]
                if v1 in vars_corr.index:  # Check if v1 is in the index
                    vars_corr.drop(v1, inplace=True)
                if len(vars_corr) > 1:
                    lst_corr = list(vars_corr.index)
                    for v2 in lst_corr:
                        if v2 not in vars2drop:
                            vars2drop.append(v2)

            X_train_processed: DataFrame = X_train.drop(vars2drop, axis=1, inplace=False)
            X_test_processed: DataFrame = X_test.drop(vars2drop, axis=1, inplace=False)

            # Evaluate the model performance with the selected variables
            eval: dict[str, list] | None = self.evaluate_step_classification(
                X_train_processed, X_test_processed, y_train, y_test,
                dataset=self.data_loader.file_tag, file_tag=f"{file_tag}_low_var_{thresh}",
                metric=metric, plot_title=f"Evaluation for {self.data_loader.file_tag} - Redundancy ({thresh})"
            )
            if eval:
                results["NB"][thresh] = eval[metric][0]
                results["KNN"][thresh] = eval[metric][1]

        nb_metrics = []
        knn_metrics = []
        for threshold in options:
            # For each threshold, get the corresponding metric value for Naive Bayes and KNN
            nb_metrics.append(results["NB"].get(threshold, None))  # Directly append the value if it's a float
            knn_metrics.append(results["KNN"].get(threshold, None))  # Same for KNN

        # Plot the results
        figure(figsize=(2 * HEIGHT, HEIGHT))
        plot_multiline_chart(
            options,
            {"NB": nb_metrics, "KNN": knn_metrics},
            title=f"{file_tag} redundancy study ({metric})",
            xlabel="correlation threshold",
            ylabel=metric,
            percentage=True,
        )
        savefig(f"graphs/classification/data_preparation/{file_tag}_fs_redundancy_{metric}_study.png")
        show()

    # %% Forecasting

    def evaluate_step_forecasting(self, X_train, X_test, y_train, y_test, file_tag):
        try:
            # Initialize the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Generate predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Validate predictions
            prd_trn = Series(y_pred_train, index=y_train.index)
            prd_tst = Series(y_pred_test, index=y_test.index)

            # Calculate evaluation metrics
            eval_results = {
                'RMSE_Train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'RMSE_Test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE_Train': mean_absolute_error(y_train, y_pred_train),
                'MAE_Test': mean_absolute_error(y_test, y_pred_test),
                'MAPE_Train': mean_absolute_percentage_error(y_train, y_pred_train),
                'MAPE_Test': mean_absolute_percentage_error(y_test, y_pred_test),
                'R2_Train': r2_score(y_train, y_pred_train),
                'R2_Test': r2_score(y_test, y_pred_test),
            }

            # Debugging Logs
            print(f"Evaluation Results: {eval_results}")

            # Plot and save results
            plot_forecasting_eval(y_train, y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - {file_tag}")
            plt.savefig(
                f"graphs/forecasting/data_preparation/{self.data_loader.file_tag}_{file_tag}_linear_regression_eval.png")
            plt.show()

            plot_forecasting_series(
                y_train,
                y_test,
                prd_tst,
                title=f"{self.data_loader.file_tag} - {file_tag}",
                xlabel=self.data_loader.read_options["index_col"],
                ylabel=self.data_loader.target,
            )
            plt.savefig(
                f"graphs/forecasting/data_preparation/{self.data_loader.file_tag}_{file_tag}_linear_regression_forecast.png")
            plt.show()

            return eval_results

        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

    def handle_missing_values_forecasting(self):
        print(f"\n\nHandling missing values for the {self.data_loader.file_tag} dataset...")

        # Categorize variables (if needed)
        variable_types = get_variable_types(self.data_loader.data)
        numeric_columns = variable_types["numeric"]

        # Prepare training and testing features/targets
        train, test = series_train_test_split(self.data_loader.data, trn_pct=0.90)
        X_train = arange(len(train)).reshape(-1, 1)
        X_test = arange(len(train), len(self.data_loader.data)).reshape(-1, 1)
        y_train = train.to_numpy()
        y_test = test.to_numpy()

        # Dictionary to store results for different techniques
        techniques = {}

        ### Mean Imputation ###
        imputer_mean_numeric = SimpleImputer(strategy='mean')

        X_train_mean = X_train.copy()
        X_test_mean = X_test.copy()

        # Apply imputation on numeric columns
        if numeric_columns:
            X_train_mean = imputer_mean_numeric.fit_transform(X_train)
            X_test_mean = imputer_mean_numeric.transform(X_test)

        # Evaluate after mean imputation
        techniques['Mean'] = self.evaluate_step_forecasting(
            X_train_mean, X_test_mean, y_train, y_test, file_tag="MV_Mean_Imputation"
        )

        ### Median Imputation ###
        imputer_median_numeric = SimpleImputer(strategy='median')

        X_train_median = X_train.copy()
        X_test_median = X_test.copy()

        # Apply imputation on numeric columns
        if numeric_columns:
            X_train_median = imputer_median_numeric.fit_transform(X_train)
            X_test_median = imputer_median_numeric.transform(X_test)

        # Evaluate after median imputation
        techniques['Median'] = self.evaluate_step_forecasting(
            X_train_median, X_test_median, y_train, y_test, file_tag="MV_Median_Imputation"
        )

        # Print techniques results
        print("\nTechniques applied and their results:")
        for technique, results in techniques.items():
            print(f"{technique}: {results}")

        return techniques

    def apply_best_missing_value_approach_forecasting(self, best_technique, techniques):
        """
        Applies the best missing value handling technique to the dataset for forecasting.
        """
        print(f"Applying the best missing value handling approach: {best_technique}")

        # Apply Mean Imputation
        if best_technique == 'Mean':
            imputer_mean_numeric = SimpleImputer(strategy='mean')

            numeric_columns = get_variable_types(self.data_loader.data.drop(columns=[self.data_loader.target]))[
                "numeric"]

            if numeric_columns:
                self.data_loader.data[numeric_columns] = imputer_mean_numeric.fit_transform(
                    self.data_loader.data[numeric_columns])

        # Apply Median Imputation
        elif best_technique == 'Median':
            imputer_median_numeric = SimpleImputer(strategy='median')

            numeric_columns = get_variable_types(self.data_loader.data.drop(columns=[self.data_loader.target]))[
                "numeric"]

            if numeric_columns:
                self.data_loader.data[numeric_columns] = imputer_median_numeric.fit_transform(
                    self.data_loader.data[numeric_columns])

        # Store the performance of the best technique for reference
        self.previous_performance = techniques[best_technique]

    def _scale_all_dataframe(self, data: DataFrame) -> DataFrame:
        vars: list[str] = data.columns.to_list()
        transf: StandardScaler = StandardScaler().fit(data)
        df = DataFrame(transf.transform(data), index=data.index)
        df.columns = vars
        return df

    def handle_scaling_forecasting(self):

        print(f"\n\nHandling scaling for the {self.data_loader.file_tag} dataset...")

        # Dictionary to store results for different techniques
        techniques = {}

        # Prepare training and testing features/targets
        train, test = dataframe_temporal_train_test_split(self.data_loader.data, trn_pct=0.90)
        X_train = train.drop(self.data_loader.target, axis=1)
        y_train = train[self.data_loader.target]
        X_test = test.drop(self.data_loader.target, axis=1)
        y_test = test[self.data_loader.target]

        # Original Dataset (Baseline)
        techniques['Original'] = self.evaluate_step_forecasting(
            X_train, X_test, y_train, y_test, file_tag="Scaling_Original"
        )

        series: Series = self.data_loader.data[self.data_loader.target]

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel=series.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} original {self.data_loader.target}",
        )
        show()

        df: DataFrame = self._scale_all_dataframe(self.data_loader.data)
        ss: Series = df[self.data_loader.target]

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            ss.index.to_list(),
            ss.to_list(),
            xlabel=ss.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} {self.data_loader.target} after scaling",
        )
        show()

        # Prepare training and testing features/targets
        train, test = dataframe_temporal_train_test_split(df, trn_pct=0.90)
        X_train = train.drop(self.data_loader.target, axis=1)
        y_train = train[self.data_loader.target]
        X_test = test.drop(self.data_loader.target, axis=1)
        y_test = test[self.data_loader.target]

        # Evaluate after mean and most frequent imputation
        techniques['Scaled'] = self.evaluate_step_forecasting(
            X_train, X_test, y_train, y_test, file_tag="Scaling"
        )

        # Print techniques results
        print("\nTechniques applied and their results:")
        for technique, results in techniques.items():
            print(f"{technique}: {results}")

        return techniques

    def apply_best_scaling_approach_forecasting(self, best_technique, techniques):

        print(f"Applying the best approach: {best_technique}")

        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        else:
            print(f"Applying Scaling...")
            self.data_loader.data = self._scale_all_dataframe(self.data_loader.data)

        self.previous_accuracy = techniques[best_technique]

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
        elif gran_level == "five_years":
            # Custom logic for five years
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df["five_years"] = (df.index.year // 5) * 5
            df = df.groupby("five_years").agg(agg_func)
            df.index = pd.to_datetime(df.index, format="%Y")  # Convert five years to timestamp
        elif gran_level == "decade":
            # Custom logic for decades
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df["decade"] = (df.index.year // 10) * 10
            df = df.groupby("decade").agg(agg_func)
            df.index = pd.to_datetime(df.index, format="%Y")  # Convert decade to timestamp
        else:
            raise ValueError(f"Unsupported granularity level: {gran_level}")

        # Convert back to Series if input was a Series
        if is_series:
            df = df.squeeze()  # Convert single-column DataFrame to Series

        return df

    def handle_aggregation_forecasting(self):
        """
        Handles aggregation for different datasets and evaluates each approach.
        """
        print(f"\n\nHandling aggregation for the {self.data_loader.file_tag} dataset...")

        # Dictionary to store results for different techniques
        techniques = {}

        # Aggregation Levels Based on Dataset Type
        if self.data_loader.file_tag == "forecast_ny_arrests":
            gran_levels = [("M", "monthly"), ("Y", "yearly")]
            original_level = "daily"
        elif self.data_loader.file_tag == "forecast_gdp_europe":
            gran_levels = [("five_years", "five_years"), ("decade", "decade")]
            original_level = "yearly"
        else:
            raise ValueError(f"Unsupported dataset")

        # Extract the target series
        series: Series = self.data_loader.data[self.data_loader.target]
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel=series.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} {original_level} {self.data_loader.target}",
        )

        # Original Dataset (Baseline)
        techniques["Original"] = self.previous_accuracy

        for gran_level, gran_name in gran_levels:
            try:
                print(f"\nTesting aggregation at {gran_name} level for {self.data_loader.file_tag}...")

                # Perform aggregation
                ss_agg: Series = self.ts_aggregation_by(series, gran_level=gran_level)

                # Visualize aggregated data
                figure(figsize=(3 * HEIGHT, HEIGHT / 2))
                plot_line_chart(
                    ss_agg.index.to_list(),
                    ss_agg.to_list(),
                    xlabel=ss_agg.index.name,
                    ylabel=self.data_loader.target,
                    title=f"{self.data_loader.file_tag} {gran_name} {self.data_loader.target}",
                )
                savefig(f"graphs/forecasting/data_preparation/{self.data_loader.file_tag}_{gran_name}_aggregation.png")

                # Aggregate dataset
                agg_data = self.ts_aggregation_by(self.data_loader.data, gran_level=gran_level)

                # Prepare training and testing features/targets
                train, test = dataframe_temporal_train_test_split(agg_data, trn_pct=0.90)
                X_train = train.drop(self.data_loader.target, axis=1)
                y_train = train[self.data_loader.target]
                X_test = test.drop(self.data_loader.target, axis=1)
                y_test = test[self.data_loader.target]

                # Evaluate the aggregated approach
                techniques[gran_name] = self.evaluate_step_forecasting(
                    X_train, X_test, y_train, y_test, file_tag=f"{gran_name}_Aggregation"
                )

            except Exception as e:
                print(f"Error during aggregation ({gran_name}): {e}")

        # Display the results of all techniques
        print("\nTechniques applied and their results:")
        for technique, results in techniques.items():
            print(f"{technique}: {results}")

        return techniques

    def apply_best_aggregation_approach_forecasting(self, best_technique, techniques):

        print(f"Applying the best approach: {best_technique}")

        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        else:
            print(f"Applying Aggregation...")

            if best_technique == "monthly":
                self.data_loader.data = self.ts_aggregation_by(self.data_loader.data, gran_level="M")

            elif best_technique == "yearly":
                self.data_loader.data = self.ts_aggregation_by(self.data_loader.data, gran_level="Y")

            elif best_technique == "five_years":
                self.data_loader.data = self.ts_aggregation_by(self.data_loader.data, gran_level="five_years")

            elif best_technique == "decade":
                self.data_loader.data = self.ts_aggregation_by(self.data_loader.data, gran_level="decade")

        self.previous_accuracy = techniques[best_technique]

    def handle_smoothing_forecasting(self, train, X_train, X_test, y_train, y_test):
        """
        Handles smoothing techniques for forecasting. Evaluates each smoothing approach
        for every defined window size and compares them. Returns aligned X_train and y_train for each smoothing size.
        """
        print(f"\n\nHandling smoothing for the {self.data_loader.file_tag} dataset...")

        # Dictionary to store results for different techniques
        techniques = {}
        smooth_data = {}  # To store aligned X_train and y_train for each smoothing size

        # Original Dataset (Baseline)
        techniques["Original"] = self.previous_accuracy

        # Define smoothing window sizes based on the dataset
        if self.data_loader.file_tag == "forecast_ny_arrests":
            sizes = [10, 25, 50, 75, 100]
        elif self.data_loader.file_tag == "forecast_gdp_europe":
            sizes = [5, 10, 20, 35, 50]
        else:
            raise ValueError(f"Unsupported dataset type: {self.data_loader.file_tag}")

        # Visualize original training data
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            train.index.to_list(),  # If train.index is the desired index
            train[self.data_loader.target].tolist(),  # Convert target column to list
            xlabel=train.index.name,
            ylabel=self.data_loader.target,
            title=f"{self.data_loader.file_tag} Original Training Data",
        )
        show()

        for size in sizes:
            print(f"\nTesting smoothing with window size={size} for {self.data_loader.file_tag}...")

            # Apply smoothing to y_train
            smoothed_y_train = y_train.rolling(window=size).mean()

            # Visualize smoothed training target
            figure(figsize=(3 * HEIGHT, HEIGHT / 2))
            plot_line_chart(
                smoothed_y_train.index.to_list(),
                smoothed_y_train.tolist(),
                xlabel=smoothed_y_train.index.name,
                ylabel=self.data_loader.target,
                title=f"{self.data_loader.file_tag} Smoothed Training Target (Window Size={size})",
            )
            show()

            # Drop NaN values caused by rolling smoothing
            smoothed_y_train = smoothed_y_train.dropna()

            # Align X_train with the smoothed target
            aligned_X_train = X_train.loc[smoothed_y_train.index]

            # Store aligned data
            smooth_data[f"Window_{size}"] = {
                "X_train": aligned_X_train,
                "y_train": smoothed_y_train,
            }

            # Evaluate the smoothed approach
            techniques[f"Smoothing_size={size}"] = self.evaluate_step_forecasting(
                aligned_X_train, X_test, smoothed_y_train, y_test, file_tag=f"Smoothing_{size}",
            )

        # Display results of all techniques
        print("\nTechniques applied and their results:")
        for technique, results in techniques.items():
            print(f"{technique}: {results}")

        return techniques, smooth_data

    def apply_best_smoothing_approach_forecasting(self, best_technique, techniques, smooth_data):
        """
        Applies the best smoothing technique to the dataset for forecasting using the aligned data.
        """
        print(f"Applying the best smoothing approach: {best_technique}")

        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        else:
            # Extract the aligned data for the best smoothing technique
            if best_technique in smooth_data:
                best_smooth_X_train = smooth_data[best_technique]["X_train"]
                best_smooth_y_train = smooth_data[best_technique]["y_train"]

                print(f"Best smoothing technique {best_technique} applied.")
                print(f"Aligned X_train shape: {best_smooth_X_train.shape}")
                print(f"Aligned y_train shape: {best_smooth_y_train.shape}")

                # Update the dataset with the aligned data
                self.X_train = best_smooth_X_train
                self.y_train = best_smooth_y_train

            else:
                raise ValueError(f"Best technique '{best_technique}' not found in aligned_data.")

        # Update the previous accuracy with the results of the best technique
        self.previous_accuracy = techniques[best_technique]

    def handle_differentiation_forecasting(self, X_train, X_test, y_train, y_test):
        """
        Handles differentiation techniques (first and second derivatives) for forecasting.
        Evaluates each technique and compares them with the original dataset.
        """
        print(f"\n\nHandling differentiation for the {self.data_loader.file_tag} dataset...")

        # Dictionary to store results for different techniques
        techniques = {}

        # Dictionary to store aligned data for different techniques
        differentiated_data = {}

        # Original Dataset (Baseline)
        techniques['Original'] = self.previous_accuracy
        differentiated_data['Original'] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }

        # First Derivative
        print(f"\nTesting the first derivative for {self.data_loader.file_tag}...")
        y_train_diff = y_train.diff().dropna()
        y_test_diff = y_test.diff().dropna()

        # Visualize the first derivative
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            y_train_diff.index, y_train_diff.values,
            title=f"{self.data_loader.file_tag} - First Derivative (Training)",
            xlabel="Index",
            ylabel=f"Δ {self.data_loader.target}",
        )
        savefig(f"graphs/forecasting/data_preparation/{self.data_loader.file_tag}_first_derivative_training.png")
        show()

        # Align X_train and X_test with y_train_diff and y_test_diff
        aligned_X_train_diff = X_train[len(X_train) - len(y_train_diff):]
        aligned_X_test_diff = X_test[len(X_test) - len(y_test_diff):]

        # Store aligned data for the first derivative
        differentiated_data['First Derivative'] = {
            "X_train": aligned_X_train_diff,
            "y_train": y_train_diff,
            "X_test": aligned_X_test_diff,
            "y_test": y_test_diff
        }

        # Evaluate the first derivative approach
        techniques['First Derivative'] = self.evaluate_step_forecasting(
            aligned_X_train_diff,
            aligned_X_test_diff,
            y_train_diff,
            y_test_diff,
            file_tag="First_Derivative"
        )

        # Second Derivative
        print(f"\nTesting the second derivative for {self.data_loader.file_tag}...")
        y_train_diff2 = y_train.diff().diff().dropna()
        y_test_diff2 = y_test.diff().diff().dropna()

        # Visualize the second derivative
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            y_train_diff2.index, y_train_diff2.values,
            title=f"{self.data_loader.file_tag} - Second Derivative (Training)",
            xlabel="Index",
            ylabel=f"Δ² {self.data_loader.target}",
        )
        savefig(f"graphs/forecasting/data_preparation/{self.data_loader.file_tag}_second_derivative_training.png")
        show()

        # Align X_train and X_test with y_train_diff2 and y_test_diff2
        aligned_X_train_diff2 = X_train[len(X_train) - len(y_train_diff2):]
        aligned_X_test_diff2 = X_test[len(X_test) - len(y_test_diff2):]

        # Store aligned data for the second derivative
        differentiated_data['Second Derivative'] = {
            "X_train": aligned_X_train_diff2,
            "y_train": y_train_diff2,
            "X_test": aligned_X_test_diff2,
            "y_test": y_test_diff2
        }

        # Evaluate the second derivative approach
        techniques['Second Derivative'] = self.evaluate_step_forecasting(
            aligned_X_train_diff2,
            aligned_X_test_diff2,
            y_train_diff2,
            y_test_diff2,
            file_tag="Second_Derivative"
        )

        # Display the results of all techniques
        print("\nTechniques applied and their results:")
        for technique, results in techniques.items():
            print(f"{technique}: {results}")

        return techniques, differentiated_data

    def apply_best_differentiation_approach_forecasting(self, best_technique, techniques, differentiated_data):
        """
        Applies the best differentiation technique to the dataset for forecasting using the aligned data.
        """
        print(f"Applying the best differentiation approach: {best_technique}")

        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        else:
            # Extract the aligned data for the best differentiation technique
            if best_technique in differentiated_data:
                best_differentiated_X_train = differentiated_data[best_technique]["X_train"]
                best_differentiated_y_train = differentiated_data[best_technique]["y_train"]
                best_differentiated_X_test = differentiated_data[best_technique]["X_test"]
                best_differentiated_y_test = differentiated_data[best_technique]["y_test"]

                print(f"Best differentiation technique {best_technique} applied.")

                # Update the dataset with the aligned data
                self.X_train = best_differentiated_X_train
                self.y_train = best_differentiated_y_train
                self.X_test = best_differentiated_X_test
                self.y_test = best_differentiated_y_test

            else:
                raise ValueError(f"Best technique '{best_technique}' not found in aligned_data.")

        # Update the previous accuracy with the results of the best technique
        self.previous_accuracy = techniques[best_technique]