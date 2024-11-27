import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import subplots, show, savefig
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from dslabs_functions import determine_outlier_thresholds_for_var


class DataProcessing:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.target = self.data_loader.target
        self.previous_accuracy = {}

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

            self.data_loader.data.drop(columns=['LAW_CODE'], inplace=True)
            print("\nDropped 'LAW_CODE' variable for being a false predictor.")
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

    def plot_technique_comparison(self, techniques, step, metric='average_accuracy'):
        """
        Plots a comparison of the techniques used in the data processing steps.

        Parameters:
        - techniques (dict): A dictionary containing the results of the techniques.
        - technique_names (list): A list of the names of the techniques.
        """
        print("\nPlotting average accuracies...")
        technique_names = list(techniques.keys())
        metric_accuracies = [techniques[tech][metric] for tech in technique_names]

        plt.figure(figsize=(8, 6))
        plt.bar(technique_names, metric_accuracies, color=['skyblue', 'orange', 'green'], alpha=0.8)
        plt.title(f'Comparison of {metric} by {step} Technique', fontsize=14)
        plt.xlabel(f'{step} Technique', fontsize=12)
        plt.ylabel(f'{metric}', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, acc in enumerate(metric_accuracies):
            plt.text(i, acc + 0.01, f"{acc:.7f}", ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()

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

        # Plot the comparison of techniques
        self.plot_technique_comparison(techniques, 'Missing Value Handling')

        # Compare techniques and choose the best one
        best_technique = max(techniques, key=lambda k: techniques[k]['average_accuracy'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Remove MV':
            self.data_loader.data = data_removed
            self.previous_accuracy = techniques['Remove MV']
        elif best_technique == 'Mean':
            self.data_loader.data = self.data_loader.data.fillna(self.data_loader.data.mean())
            self.previous_accuracy = techniques['Mean']
        elif best_technique == 'Median':
            self.data_loader.data = self.data_loader.data.fillna(self.data_loader.data.median())
            self.previous_accuracy = techniques['Median']

        print("Missing value handling completed.")

    def handle_outliers(self):
        """
        Handles outliers using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Original Dataset (no outlier removal)
        - Replacing Outliers
        - Truncating Outliers
        """

        print("\nHandling outliers...")

        # Dictionary to store results for different techniques
        techniques = {}

        # Original Dataset (Baseline)
        print("\nEvaluating Original Dataset (No Outlier Removal)...")
        print(f"Original Dataset performance: {self.previous_accuracy}")
        print("Original data shape:", self.data_loader.data.shape)
        techniques['Original'] = self.previous_accuracy

        X = self.data_loader.data.drop(columns=[self.target])
        numeric_vars = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_vars:
            print("There are no numeric variables to process for outliers.")
            return

        # Drop Outliers
        print("\nEvaluating Drop Outliers...")
        df_dropped = self.data_loader.data.copy()
        summary = df_dropped[numeric_vars].describe()

        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary[var])
            outliers = df_dropped[(df_dropped[var] > top) | (df_dropped[var] < bottom)]
            df_dropped.drop(outliers.index, axis=0, inplace=True)

        results_drop = self.evaluate_step(df_dropped.drop(columns=[self.target]), df_dropped[self.target])
        print(f"Drop Outliers performance: {results_drop}")
        print("Data shape after dropping outliers:", df_dropped.shape)
        techniques['Drop'] = results_drop

        # Replace Outliers with Median
        print("\nEvaluating Replace Outliers...")
        df_replaced = self.data_loader.data.copy()

        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary[var])
            median = df_replaced[var].median()
            df_replaced[var] = df_replaced[var].apply(lambda x: median if x > top or x < bottom else x)

        results_replace = self.evaluate_step(df_replaced.drop(columns=[self.target]), df_replaced[self.target])
        print(f"Replace Outliers performance: {results_replace}")
        print("Data shape after replacing outliers:", df_replaced.shape)
        print("Data description after replacing outliers:\n", df_replaced.describe())
        techniques['Replace'] = results_replace

        # Truncate Outliers
        print("\nEvaluating Truncate Outliers...")
        df_truncated = self.data_loader.data.copy()

        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary[var])
            df_truncated[var] = df_truncated[var].apply(lambda x: top if x > top else (bottom if x < bottom else x))

        results_truncate = self.evaluate_step(df_truncated.drop(columns=[self.target]), df_truncated[self.target])
        print(f"Truncate Outliers performance: {results_truncate}")
        print("Data shape after truncating outliers:", df_truncated.shape)
        print("Data description after truncating outliers:\n", df_truncated.describe())
        techniques['Truncate'] = results_truncate

        # Plot the comparison of techniques
        self.plot_technique_comparison(techniques, 'Outlier Handling')

        # Select the best technique
        best_technique = max(techniques, key=lambda k: techniques[k]['average_accuracy'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        elif best_technique == 'Drop':
            self.data_loader.data = df_dropped
            self.previous_accuracy = techniques['Drop']
        elif best_technique == 'Replace':
            self.data_loader.data = df_replaced
            self.previous_accuracy = techniques['Replace']
        elif best_technique == 'Truncate':
            self.data_loader.data = df_truncated
            self.previous_accuracy = techniques['Truncate']

        print("\nOutlier handling completed.")

    def handle_scaling(self):
        """
        Handles scaling using different techniques and selects the best-performing one.
        The evaluated techniques are:
        - Standard Scaler
        - MinMax Scaler
        """

        print("\nHandling scaling...")

        # List to store results for different techniques
        techniques = {}

        # Original Dataset (Baseline)
        print("\nEvaluating Original Dataset (No scaling applied)...")
        print(f"Original Dataset performance: {self.previous_accuracy}")
        techniques['Original'] = self.previous_accuracy

        target_data: Series = self.data_loader.data.pop(self.target)
        vars: list[str] = self.data_loader.data.columns.to_list()
        vars.append(self.target)

        # Standard Scaler
        transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(self.data_loader.data)
        df_zscore = DataFrame(transf.transform(self.data_loader.data), index=self.data_loader.data.index)
        df_zscore[self.target] = target_data
        df_zscore.columns = vars
        df_zscore.to_csv(f"data/{self.data_loader.file_tag}_scaled_zscore.csv", index=False)

        print("\nEvaluating Standard Scaler...")
        results_standard = self.evaluate_step(df_zscore.drop(columns=[self.target]), df_zscore[self.target])
        print(f"Standard Scaler performance: {results_standard}")
        techniques['Standard'] = results_standard

        # MinMax Scaler
        transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(self.data_loader.data)
        df_minmax = DataFrame(transf.transform(self.data_loader.data), index=self.data_loader.data.index)
        df_minmax[self.target] = target_data
        df_minmax.columns = vars
        df_minmax.to_csv(f"data/{self.data_loader.file_tag}_scaled_minmax.csv", index=False)

        print("\nEvaluating MinMax Scaler...")
        results_minmax = self.evaluate_step(df_minmax.drop(columns=[self.target]), df_minmax[self.target])
        print(f"MinMax Scaler performance: {results_minmax}")
        techniques['MinMax'] = results_minmax

        # Plot the comparison of techniques
        self.plot_technique_comparison(techniques, 'Scaling Handling', 'knn')

        # Compare techniques and choose the best one
        best_technique = max(techniques, key=lambda k: techniques[k]['knn'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Original':
            print("No changes made to the dataset (original data retained).")
        elif best_technique == 'Standard':
            self.data_loader.data = df_zscore
            self.previous_accuracy = techniques['Standard']
        elif best_technique == 'MinMax':
            self.data_loader.data = df_minmax
            self.previous_accuracy = techniques['MinMax']

        print("Scaling handling completed.")

        fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
        axs[0, 1].set_title("Original data")
        self.data_loader.data.boxplot(ax=axs[0, 0])
        axs[0, 0].set_title("Z-score normalization")
        df_zscore.boxplot(ax=axs[0, 1])
        axs[0, 2].set_title("MinMax normalization")
        df_minmax.boxplot(ax=axs[0, 2])
        # savefig(f"graphs/data_processing/data_scaling/{self.data_loader.file_tag}_different_scaler_comparison.png")
        show()

    def handle_feature_selection(self):
        """
        Performs feature selection using:
        - Dropping Low Variance Variables
        - Dropping Redundant Variables (via correlation)
        The best dataset configuration is selected based on model performance.
        Tracks and reports the selected and dropped variables.
        """
        print("\nStarting feature selection...")

        X = self.data_loader.data.drop(columns=[self.target])
        y = self.data_loader.data[self.target]

        techniques = {}
        dropped_features = {}

        # Original Dataset (Baseline)
        print("\nEvaluating Original Dataset (No Feature Selection)...")
        results_original = self.evaluate_step(X, y)
        print(f"Original Dataset performance: {results_original}")
        techniques['Original'] = results_original
        dropped_features['Original'] = []

        # Dropping Low Variance Variables
        print("\nEvaluating Low Variance Feature Selection...")
        low_variance_filter = VarianceThreshold(threshold=0.01)  # Adjust threshold as necessary
        low_variance_filter.fit(X)  # Fit the filter to the data
        low_variance_support = low_variance_filter.get_support()  # Get the mask for retained features
        X_low_variance = pd.DataFrame(
            low_variance_filter.transform(X),  # Use transform to apply the filter
            columns=X.columns[low_variance_support]
        )

        dropped_low_variance = X.columns[~low_variance_support].tolist()
        dropped_features['Low Variance'] = dropped_low_variance

        if not X_low_variance.empty:
            results_low_variance = self.evaluate_step(X_low_variance, y)
            print(f"Low Variance Feature Selection performance: {results_low_variance}")
            print(f"Dropped Low Variance Variables: {dropped_low_variance}")
            techniques['Low Variance'] = results_low_variance
        else:
            print("Low Variance Feature Selection resulted in an empty dataset.")
            techniques['Low Variance'] = {'knn': 0, 'nb': 0, 'average_accuracy': 0}

        # Dropping Redundant Variables (via correlation)
        print("\nEvaluating Redundant Feature Removal...")
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        redundant_features = [column for column in upper_triangle.columns if
                              any(upper_triangle[column] > 0.95)]  # Threshold = 0.95
        X_redundant_removed = X.drop(columns=redundant_features)

        dropped_features['Redundant'] = redundant_features

        if not X_redundant_removed.empty:
            results_redundant = self.evaluate_step(X_redundant_removed, y)
            print(f"Redundant Feature Removal performance: {results_redundant}")
            print(f"Dropped Redundant Variables: {redundant_features}")
            techniques['Redundant'] = results_redundant
        else:
            print("Redundant Feature Removal resulted in an empty dataset.")
            techniques['Redundant'] = {'knn': 0, 'nb': 0, 'average_accuracy': 0}

        # Compare techniques and choose the best one
        best_technique = max(techniques, key=lambda k: techniques[k]['average_accuracy'])
        print(f"\nBest technique: {best_technique} with accuracy: {techniques[best_technique]}")

        # Apply the best technique to the dataset
        if best_technique == 'Original':
            print("No changes made to the dataset (original features retained).")
            selected_features = X.columns.tolist()
        elif best_technique == 'Low Variance':
            self.data_loader.data = pd.concat([X_low_variance, y], axis=1)
            selected_features = X_low_variance.columns.tolist()
        elif best_technique == 'Redundant':
            self.data_loader.data = pd.concat([X_redundant_removed, y], axis=1)
            selected_features = X_redundant_removed.columns.tolist()

        print(f"\nSelected Features: {selected_features}")
        print(f"Dropped Features for {best_technique}: {dropped_features[best_technique]}")
        print("Feature selection completed.")