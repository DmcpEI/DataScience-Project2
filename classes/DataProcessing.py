from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, Series, Index
from matplotlib.pyplot import subplots, show, savefig, figure
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dslabs_functions import determine_outlier_thresholds_for_var, run_NB, run_KNN, CLASS_EVAL_METRICS, \
    plot_multibar_chart, get_variable_types, concat, plot_multiline_chart, HEIGHT
from imblearn.over_sampling import SMOTE
from numpy import ndarray


class DataProcessing:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.target = self.data_loader.target
        self.previous_accuracy = {}
        self.X_train = DataFrame()
        self.X_test = DataFrame()
        self.y_train = Series()
        self.y_test = Series()

    def pre_encode_variables(self):

        # Encode AGE_GROUP
        print("Encoding AGE_GROUP...")
        valid_age_groups = {'<18', '18-24', '25-44', '45-64', '65+'}
        self.data_loader.data['AGE_GROUP'] = self.data_loader.data['AGE_GROUP'].where(
            self.data_loader.data['AGE_GROUP'].isin(valid_age_groups), 'UNKNOWN')
        age_group_mapping = {'UNKNOWN': None, '<18': 1, '18-24': 2, '25-44': 3, '45-64': 4, '65+': 5}
        self.data_loader.data['AGE_GROUP'] = self.data_loader.data['AGE_GROUP'].map(age_group_mapping)

        # Encode JURISDICTION_CODE
        self.data_loader.data['JURISDICTION_CODE'] = self.data_loader.data['JURISDICTION_CODE'].apply(lambda x: 'NY' if x < 3 else 'nonNY')

    def encode_variables(self):
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
        from sklearn.feature_extraction.text import CountVectorizer

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
        label_encoder = LabelEncoder()
        self.data_loader.data['LAW_CODE'] = label_encoder.fit_transform(
            self.data_loader.data['LAW_CODE'].fillna('UNKNOWN'))

        # Encode LAW_CAT_CD, ARREST_BORO, PERP_SEX, PERP_RACE
        print("Encoding other variables...")
        mapping = {
            'JURISDICTION_CODE': {'NY': 1, 'nonNY': 0},
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

    def drop_variables(self):
        """
        Drop variables that are false predictions or irrelevant
        """
        if self.data_loader.target == "LAW_CAT_CD":

            self.data_loader.data.drop(columns=['ARREST_KEY'], inplace=True)
            print("\nDropped 'ARREST_KEY' variable for being irrelevant for the classification task.")

            # self.data_loader.data.drop(columns=['PD_DESC'], inplace=True)
            # print("\nDropped 'PD_DESC' variable for being irrelevant for the classification task.")
            #
            # self.data_loader.data.drop(columns=['OFNS_DESC'], inplace=True)
            # print("\nDropped 'OFNS_DESC' variable for being irrelevant for the classification task.")

            # self.data_loader.data.drop(columns=['PD_CD'], inplace=True)
            # print("\nDropped 'PD_CD' variable for being a false predictor.")
            #
            # self.data_loader.data.drop(columns=['KY_CD'], inplace=True)
            # print("\nDropped 'KY_CD' variable for being a false predictor.")

            # self.data_loader.data.drop(columns=['LAW_CODE'], inplace=True)
            # print("Dropped 'LAW_CODE' variable for being a false predictor.")

        elif self.data_loader.target == "CLASS":

            self.data_loader.data.drop(columns=['Financial Distress'], inplace=True)
            print("\nDropped 'Financial Distress' variable for being a false predictor.")

    def evaluate_step(self, X_train, X_test, y_train, y_test, dataset, file_tag, metric="accuracy", plot_title="Model Evaluation", is_scalling = False):
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
            savefig(f"graphs/data_preparation/{dataset}_{file_tag}_eval.png")
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
            savefig(f"graphs/data_preparation/{dataset}_{file_tag}_eval.png")
            show()

            return eval_results

    def handle_missing_values(self):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Dictionary to store results
        techniques = {}

        # Row Removal
        data_removed = self.data_loader.data.dropna()

        if not data_removed.empty:  # Ensure removal doesn't leave the dataset empty
            X_removed = data_removed.drop(columns=[self.target])
            y_removed = data_removed[self.target]
            X_train_removed, X_test_removed, y_train_removed, y_test_removed = train_test_split(
                X_removed, y_removed, test_size=0.2, random_state=42
            )
            results_removal = self.evaluate_step(
                X_train_removed, X_test_removed, y_train_removed, y_test_removed,
                self.data_loader.file_tag, "MV_Row_Removal",
                plot_title=f"Evaluation for {self.data_loader.file_tag} - MV Row Removal"
            )
            techniques['Remove MV'] = results_removal
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

        results_mean_most_frequent = self.evaluate_step(
            X_train_mean, X_test_mean, y_train, y_test,
            self.data_loader.file_tag, "MV_Mean_Most_Frequent_Imputation",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - MV Mean & Most Frequent Imputation"
        )
        techniques['Mean & Most Frequent'] = results_mean_most_frequent

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

        results_median_most_frequent = self.evaluate_step(
            X_train_median, X_test_median, y_train, y_test,
            self.data_loader.file_tag, "MV_Median_Most_Frequent_Imputation",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - MV Median & Most Frequent Imputation"
        )
        techniques['Median & Most Frequent'] = results_median_most_frequent

        # Print all performances at the end
        print("\nSummary of Missing Value Handling Techniques:")
        for technique, performance in techniques.items():
            print(f"{technique} performance: {performance}")

        return techniques

    def apply_best_missing_value_approach(self, best_technique, techniques):
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

    def handle_outliers(self):
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

        results_drop = self.evaluate_step(df_train_dropped.drop(columns=[self.target]), self.X_test,
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

        results_replace = self.evaluate_step(df_train_replaced.drop(columns=[self.target]), self.X_test,
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

        results_truncate = self.evaluate_step(df_train_truncated.drop(columns=[self.target]), self.X_test,
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

    def apply_best_outliers_approach(self, approach, techniques, X_train = None, y_train = None):

        numeric_vars = self.X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_vars:
            print("There are no numeric variables to process for outliers.")
            return
        match approach:
            case 'Original':
                print("No changes made to the dataset (original data retained).")
            case 'Drop':
                print("Applying Drop Outliers...")
                self.X_train = X_train
                self.y_train = y_train
            case 'Replace':
                print("Applying Replace Outliers...")
                self.X_train = X_train
                self.y_train = y_train
            case 'Truncate':
                print("Applying Truncate Outliers...")
                self.X_train = X_train
                self.y_train = y_train

        self.previous_accuracy = techniques[approach]

    def handle_scaling(self):
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
        results_standard = self.evaluate_step(
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
        results_minmax = self.evaluate_step(
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

    def apply_best_scaling_approach(self, best_technique, techniques, X_train = None, y_train = None, X_test = None, y_test = None):
        """
        Applies the best scaling technique to the dataset.
        """
        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        elif best_technique == 'Standard':
            print("Applying Standard Scaler...")
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        elif best_technique == 'MinMax':
            print("Applying MinMax Scaler...")
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

        self.previous_accuracy = techniques[best_technique]

    def handle_balancing(self):
        """
        Handles class imbalance using undersampling, oversampling, and SMOTE techniques,
        and evaluates the performance of each.
        """
        print(f"\n\nHandling balancing for the {self.data_loader.file_tag} dataset...\n")

        # Combine training data and target for easier processing
        data_train = pd.concat([self.X_train, self.y_train], axis=1)

        # Ensure no duplicate columns
        data_train = data_train.loc[:, ~data_train.columns.duplicated()]

        # Get class counts and identify minority and majority classes
        target_count = data_train[self.target].value_counts()
        positive_class = target_count.idxmin()
        negative_class = target_count.idxmax()

        print(f"Class distribution before balancing: {target_count.to_dict()}")

        techniques = {"Original": self.previous_accuracy}

        # Random Undersampling
        df_positives = data_train[data_train[self.target] == positive_class]
        df_negatives = data_train[data_train[self.target] == negative_class].sample(len(df_positives), random_state=42)
        df_under = pd.concat([df_positives, df_negatives], axis=0)
        X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(
            df_under.drop(columns=[self.target]), df_under[self.target], test_size=0.3, random_state=42
        )
        techniques["Undersampling"] = self.evaluate_step(
            X_train_under, X_test_under, y_train_under, y_test_under, self.data_loader.file_tag,
            "Balancing_Undersampling",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - Undersampling"
        )

        # Random Oversampling
        df_negatives = data_train[data_train[self.target] == negative_class]
        df_positives = data_train[data_train[self.target] == positive_class].sample(len(df_negatives), replace=True,
                                                                                    random_state=42)
        df_over = pd.concat([df_positives, df_negatives], axis=0)
        X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(
            df_over.drop(columns=[self.target]), df_over[self.target], test_size=0.3, random_state=42
        )
        techniques["Oversampling"] = self.evaluate_step(
            X_train_over, X_test_over, y_train_over, y_test_over, self.data_loader.file_tag,
            "Balancing_Oversampling",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - Oversampling"
        )

        # SMOTE
        smote = SMOTE(sampling_strategy="minority", random_state=42)
        X = data_train.drop(columns=[self.target])
        y = data_train[self.target]
        smote_X, smote_y = smote.fit_resample(X, y)
        X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
            smote_X, smote_y, test_size=0.3, random_state=42
        )
        techniques["SMOTE"] = self.evaluate_step(
            X_train_smote, X_test_smote, y_train_smote, y_test_smote, self.data_loader.file_tag,
            "Balancing_SMOTE",
            plot_title=f"Evaluation for {self.data_loader.file_tag} - SMOTE"
        )

        # Print all performances at the end
        print("\nSummary of Balancing Techniques:")
        for technique, performance in techniques.items():
            print(f"{technique} performance: {performance}")

        return techniques, df_under, df_over, smote_X, smote_y

    def apply_best_balancing_approach(self, best_technique, techniques, X_train = None, y_train = None):
        """
        Applies the best balancing technique to the dataset.
        """
        match best_technique:
            case 'Original':
                print("No changes made to the dataset (original data retained).")
            case 'Undersampling':
                print("Applying Random Undersampling...")
                self.X_train = X_train
                self.y_train = y_train
            case 'Oversampling':
                print("Applying Random Oversampling...")
                self.X_train = X_train
                self.y_train = y_train
            case 'SMOTE':
                print("Applying SMOTE...")
                self.X_train = X_train
                self.y_train = y_train

        self.previous_accuracy = techniques[best_technique]

    def handle_feature_selection(self, X_train, X_test, y_train, y_test):

        vars_to_drop = self.select_low_variance_variables(max_threshold=3)
        print(f"Variables to drop: {vars_to_drop}")
        self.study_variance(X_train, X_test, y_train, y_test, max_threshold=1, lag=0.05, metric="accuracy",
                            file_tag="ny_arrests_variance")
        # self.apply_feature_selection(vars_to_drop, file_tag="ny_arrests_feature_selection")

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
            eval: dict[str, list] | None = self.evaluate_step(
                X_train_processed, X_test_processed, y_train, y_test,
                dataset=self.data_loader.file_tag, file_tag=f"{file_tag}_low_var_{thresh}",
                metric=metric, plot_title=f"Evaluation for {self.data_loader.file_tag} - Low Variance ({thresh})"
            )
            if eval:
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
        savefig(f"graphs/{file_tag}_fs_low_var_{metric}_study.png")
        show()

    def apply_feature_selection(self, vars2drop: list[str], file_tag: str):
        """
        Applies the feature selection to training and testing datasets.
        :param vars2drop: List of variables to drop.
        """
        train = self.data_loader.data.drop(columns=[self.target])
        test = self.data_loader.data[self.target]

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)

        # Save processed files
        train_copy.to_csv(f"data/{file_tag}_train_lowvar.csv", index=True)
        test_copy.to_csv(f"data/{file_tag}_test_lowvar.csv", index=True)

        # Update the data loader
        self.data_loader.train = train_copy
        self.data_loader.test = test_copy

        print(f"Feature selection applied. Train shape: {train_copy.shape}, Test shape: {test_copy.shape}")