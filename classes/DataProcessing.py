from math import ceil

import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame, Series, Index
from matplotlib.pyplot import subplots, show, figure, savefig
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from dslabs_functions import plot_multiline_chart, HEIGHT


class DataProcessing:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data = self.data_loader.data
        self.target = self.data_loader.target

    def encode_variables(self):
        """
        Encodes all symbolic variables in the dataset based on the granularity analysis.
        """

        # Encode AGE_GROUP
        print("Encoding AGE_GROUP...")
        valid_age_groups = {'<18', '18-24', '25-44', '45-64', '65+'}
        self.data_loader.data['AGE_GROUP'] = self.data_loader.data['AGE_GROUP'].where(self.data_loader.data['AGE_GROUP'].isin(valid_age_groups), 'UNKNOWN')
        age_group_mapping = {'UNKNOWN': 0, '<18': 1, '18-24': 2, '25-44': 3, '45-64': 4, '65+': 5}
        self.data_loader.data['AGE_GROUP'] = self.data_loader.data['AGE_GROUP'].map(age_group_mapping)

        # Encode ARREST_DATE
        print("Encoding ARREST_DATE...")
        arrest_date = pd.to_datetime(self.data_loader.data['ARREST_DATE'], format='%m/%d/%Y', errors='coerce')
        self.data_loader.data['ARREST_DAY'] = arrest_date.dt.day
        self.data_loader.data['ARREST_MONTH'] = arrest_date.dt.month
        self.data_loader.data['ARREST_YEAR'] = arrest_date.dt.year
        self.data_loader.data['ARREST_DAYOFWEEK'] = arrest_date.dt.dayofweek
        self.data_loader.data.drop(columns=['ARREST_DATE'], inplace=True)

        # Cyclic Encoding for ARREST_DAYOFWEEK
        print("Applying cyclic encoding to ARREST_DAYOFWEEK...")
        self.data_loader.data['ARREST_DAYOFWEEK_SIN'] = np.sin(2 * np.pi * self.data_loader.data['ARREST_DAYOFWEEK'] / 7)
        self.data_loader.data['ARREST_DAYOFWEEK_COS'] = np.cos(2 * np.pi * self.data_loader.data['ARREST_DAYOFWEEK'] / 7)
        self.data_loader.data.drop(columns=['ARREST_DAYOFWEEK'], inplace=True)  # Drop original column

        # Encode PD_DESC
        print("Encoding PD_DESC...")
        self.data_loader.data['PD_DESC'] = self.data_loader.data['PD_DESC'].fillna("UNKNOWN").apply(
            lambda desc: 3 if len(desc) > 40 else 2 if len(desc) > 30 else 1 if len(desc) > 20 else 0
        )

        # Encode OFNS_DESC
        print("Encoding OFNS_DESC...")
        label_encoder = LabelEncoder()
        self.data_loader.data['OFNS_DESC'] = label_encoder.fit_transform(self.data_loader.data['OFNS_DESC'].fillna('UNKNOWN'))

        # Encode LAW_CODE
        print("Encoding LAW_CODE...")
        label_encoder = LabelEncoder()
        self.data_loader.data['LAW_CODE'] = label_encoder.fit_transform(self.data_loader.data['LAW_CODE'].fillna('UNKNOWN'))

        # Encode LAW_CAT_CD, ARREST_BORO, PERP_SEX, PERP_RACE
        print("Encoding other variables...")
        mapping = {
            'LAW_CAT_CD': {'F': 1, 'M': 0},
            'ARREST_BORO': {'M': 1, 'B': 2, 'Q': 3, 'K': 4, 'S': 5},
            'PERP_SEX': {'M': 1, 'F': 0},
            'PERP_RACE': {
                'UNKNOWN': 0, 'BLACK': 1, 'BLACK HISPANIC': 2, 'ASIAN / PACIFIC ISLANDER': 3,
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

            self.data_loader.data.drop(columns=['PD_CD'], inplace=True)
            print("\nDropped 'PD_CD' variable for being highly correlated with the 'KY_CD' variable.")

            self.data_loader.data.drop(columns=['PD_DESC'], inplace=True)
            print("\nDropped 'PD_DESC' variable for being highly correlated with the 'OFNS_DESC' variable.")
        elif self.data_loader.target == "CLASS":

            self.data_loader.data.drop(columns=['Financial Distress'], inplace=True)
            print("\nDropped 'Financial Distress' variable for being a false predictor.")

    def evaluate_step(self, X, y, test_size=0.3):
        """
        Evaluates the model's performance on a given dataset.

        Parameters:
        - X (pd.DataFrame): The dataset to evaluate.
        - y (pd.Series): The target variable.
        - test_size (float): The proportion of the dataset to include in the test split.

        Returns:
        - results (dict): A dictionary containing model performance metrics.
        """
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        results = {}

        # KNN Model
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
        results['knn'] = knn_accuracy

        # Naive Bayes Model
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        nb_accuracy = accuracy_score(y_test, nb.predict(X_test))
        results['nb'] = nb_accuracy

        # Calculate the average accuracy
        results['average_accuracy'] = (results['knn'] + results['nb']) / 2

        return results

    def handle_missing_values(self):
        """
        Handles missing values using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Mean Imputation
        - Median Imputation
        - Row Removal (drops rows with any missing values)
        """

        X = self.data_loader.data.drop(columns=[self.target])
        y = self.data_loader.data[self.target]

        print("\nHandling missing values...")

        # List to store results
        techniques = {}

        # Row Removal
        print("\nEvaluating Row Removal...")
        data_removed = self.data_loader.data.dropna()

        if not data_removed.empty:  # Ensure removal doesn't leave the dataset empty
            X_removed = data_removed.drop(columns=[self.target])
            y_removed = data_removed[self.target]
            results_removal = self.evaluate_step(X_removed, y_removed)
            print(f"Row Removal performance: {results_removal}")
            techniques['Remove MV'] = results_removal
        else:
            print("\nRow Removal skipped (would result in an empty dataset).")
            techniques['Remove MV'] = {'knn': 0, 'nb': 0, 'average_accuracy': 0}

        # Mean Imputation
        print("\nEvaluating Mean Imputation...")
        imputer_mean = SimpleImputer(strategy='mean')
        X_mean_imputed = pd.DataFrame(imputer_mean.fit_transform(X), columns=X.columns)
        results_mean = self.evaluate_step(X_mean_imputed, y)
        print(f"Mean Imputation performance: {results_mean}")
        techniques['Mean'] = results_mean

        # Median Imputation
        print("\nEvaluating Median Imputation...")
        imputer_median = SimpleImputer(strategy='median')
        X_median_imputed = pd.DataFrame(imputer_median.fit_transform(X), columns=X.columns)
        results_median = self.evaluate_step(X_median_imputed, y)
        print(f"Median Imputation performance: {results_median}")
        techniques['Median'] = results_median

        # Compare techniques and choose the best one
        best_technique = max(techniques, key=lambda k: techniques[k]['average_accuracy'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Remove MV':
            self.data_loader.data = data_removed
        elif best_technique == 'Mean':
            self.data_loader.data = self.data_loader.data.fillna(self.data_loader.data.mean())
        elif best_technique == 'Median':
            self.data_loader.data = self.data_loader.data.fillna(self.data_loader.data.median())

        print("Missing value handling completed.")

    def handle_outliers(self):
        """
        Handles outliers using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Original Dataset (no outlier removal)
        - Z-Score Removal
        - IQR Removal
        """

        X = self.data_loader.data.drop(columns=[self.target])
        y = self.data_loader.data[self.target]

        print("\nHandling outliers...")

        # List to store results for different techniques
        techniques = {}

        # Original Dataset (Baseline)
        print("\nEvaluating Original Dataset (No Outlier Removal)...")
        results_original = self.evaluate_step(X, y)
        print(f"Original Dataset performance: {results_original}")
        print(f"Original Dataset shape: {X.shape}")
        techniques['Original'] = results_original

        # Z-Score Method
        print("\nEvaluating Z-Score Removal...")
        zscore = lambda x: (x - x.mean()) / x.std()
        X_zscore_removed = X[(zscore(X) <= 3).all(axis=1)]
        y_zscore_removed = y[X_zscore_removed.index]  # Keep target aligned

        if not X_zscore_removed.empty:
            results_zscore = self.evaluate_step(X_zscore_removed, y_zscore_removed)
            print(f"Z-Score Removal performance: {results_zscore}")
            print(f"Z-Score Removal shape: {X_zscore_removed.shape}")
            techniques['Z-Score'] = results_zscore
        else:
            print("\nZ-Score Removal skipped (would result in an empty dataset).")
            techniques['Z-Score'] = {'knn': 0, 'nb': 0, 'average_accuracy': 0}

        # IQR Method
        print("\nEvaluating IQR Removal...")
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        X_iqr_removed = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        y_iqr_removed = y[X_iqr_removed.index]  # Keep target aligned

        if not X_iqr_removed.empty:
            results_iqr = self.evaluate_step(X_iqr_removed, y_iqr_removed)
            print(f"IQR Removal performance: {results_iqr}")
            print(f"IQR Removal shape: {X_iqr_removed.shape}")
            techniques['IQR'] = results_iqr
        else:
            print("\nIQR Removal skipped (would result in an empty dataset).")
            techniques['IQR'] = {'knn': 0, 'nb': 0, 'average_accuracy': 0}

        # Replacing outliers with fixed values /// Truncating outliers

        # Compare techniques and choose the best one
        best_technique = max(techniques, key=lambda k: techniques[k]['average_accuracy'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        elif best_technique == 'Z-Score':
            self.data_loader.data = pd.concat([X_zscore_removed, y_zscore_removed], axis=1)
        elif best_technique == 'IQR':
            self.data_loader.data = pd.concat([X_iqr_removed, y_iqr_removed], axis=1)

        print("Outlier handling completed.")

    def handle_scaling(self):
        """
        Handles scaling using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Standard Scaler
        - MinMax Scaler
        """
        data = self.data_loader.data
        target = self.data_loader.target

        X = self.data_loader.data.drop(columns=[self.target])
        y = self.data_loader.data[self.target]

        target_data: Series = data.pop(target)
        vars: list[str] = data.columns.to_list()
        vars.append(target)

        print("\nHandling scaling...")

        # List to store results for different techniques
        techniques = {}

        # TODO: Calculate KNN for original data to check if scaling improves or not

        # Standard Scaler
        transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
        df_zscore = DataFrame(transf.transform(data), index=data.index)
        df_zscore[target] = target_data
        df_zscore.columns = vars

        results_standard = self.evaluate_step(df_zscore.drop(columns=[target]), df_zscore[target])
        print(f"Standard Scaler performance: {results_standard}")
        techniques['Standard'] = results_standard

        # MinMax Scaler
        transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
        df_minmax = DataFrame(transf.transform(data), index=data.index)
        df_minmax[target] = target_data
        df_minmax.columns = vars

        results_minmax = self.evaluate_step(df_minmax.drop(columns=[target]), df_minmax[target])
        print(f"MinMax Scaler performance: {results_minmax}")
        techniques['MinMax'] = results_minmax

        # Compare techniques and choose the best one
        best_technique = max(techniques, key=lambda k: techniques[k]['knn'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Standard':
            self.data_loader.data = df_zscore
        elif best_technique == 'MinMax':
            self.data_loader.data = df_minmax

        print("Scaling handling completed.")

        fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
        axs[0, 1].set_title("Original data")
        data.boxplot(ax=axs[0, 0])
        axs[0, 0].set_title("Z-score normalization")
        df_zscore.boxplot(ax=axs[0, 1])
        axs[0, 2].set_title("MinMax normalization")
        df_minmax.boxplot(ax=axs[0, 2])
        show()


    def handle_feature_selection(self):
        #split the dataset into train and test
        X = self.data_loader.data.drop(columns=[self.target])
        y = self.data_loader.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vars_to_drop = self.select_low_variance_variables(max_threshold=3)
        print(f"Variables to drop: {vars_to_drop}")
        self.study_variance(X_train, X_test, y_train, y_test, max_threshold=3, lag=0.1, metric="accuracy", file_tag="ny_arrests_variance")
        #self.apply_feature_selection(vars_to_drop, file_tag="ny_arrests_feature_selection")

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

    def study_variance(self, X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame, max_threshold: float = 1, lag: float = 0.05,
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
            #if the target is in the variables to drop, remove it from that list
            if self.target in vars2drop:
                vars2drop = vars2drop.drop(self.target)

            # Drop the low variance variables from the training and testing datasets
            X_train_processed: DataFrame = X_train.drop(vars2drop, axis=1, inplace=False)
            X_test_processed: DataFrame = X_test.drop(vars2drop, axis=1, inplace=False)

            X_combined = pd.concat([X_train_processed, X_test_processed], axis=0)
            y_combined = pd.concat([y_train, y_test], axis=0)

            # Evaluate the model performance with the selected variables
            eval: dict[str, list] | None = self.evaluate_step(
                X_combined, y_combined
            )
            if eval:
                results["NB"][thresh] = eval['nb']
                results["KNN"][thresh] = eval['knn']

            print(results)

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
