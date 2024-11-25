import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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
                'UNKNOWN': 0, 'BLACK': 1, 'WHITE HISPANIC': 2, 'WHITE': 3,
                'BLACK HISPANIC': 4, 'ASIAN / PACIFIC ISLANDER': 5,
                'AMERICAN INDIAN/ALASKAN NATIVE': 6, 'OTHER': 7
            }
        }

        for col, mapping_dict in mapping.items():
            self.data_loader.data[col] = self.data_loader.data[col].map(mapping_dict)

        print("All symbolic variables encoded.")

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