import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataProcessing:

    def __init__(self, data_loader):

        self.data_loader = data_loader
        self.data = self.data_loader.data
        self.target = self.data_loader.target

    def group_AGE_GROUP(self):
        """
        Group the AGE_GROUP outliers into 'UNKNOWN'.
        """
        # Define valid AGE_GROUP values
        valid_age_groups = {'<18', '18-24', '25-44', '45-64', '65+'}
        # Replace outliers with 'UNKNOWN'
        self.data['AGE_GROUP'] = self.data['AGE_GROUP'].where(self.data['AGE_GROUP'].isin(valid_age_groups), 'UNKNOWN')

    def _encode_ARREST_DATE(self):
        """Encode the ARREST_DATE variable"""
        # Convert ARREST_DATE to datetime format (assuming the format is mm/dd/yyyy)
        self.data['ARREST_DATE'] = pd.to_datetime(self.data['ARREST_DATE'], format='%m/%d/%Y', errors='coerce')

        # Extract useful features from the ARREST_DATE
        self.data['ARREST_DAY'] = self.data['ARREST_DATE'].dt.day  # Day of the month (1-31)
        self.data['ARREST_MONTH'] = self.data['ARREST_DATE'].dt.month  # Month (1-12)
        self.data['ARREST_YEAR'] = self.data['ARREST_DATE'].dt.year  # Year (e.g., 2020)
        self.data['ARREST_DAYOFWEEK'] = self.data['ARREST_DATE'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)

        self.data.drop(columns=['ARREST_DATE'], inplace=True)

    def _encode_PD_DESC(self):
        """Encode the PD_DESC variable with numerical severity encoding"""
        # Handle missing values in PD_DESC and replace with "UNKNOWN"
        self.data["PD_DESC"] = self.data["PD_DESC"].fillna("UNKNOWN")

        # Apply severity categorization based on string length directly to PD_DESC
        self.data["PD_DESC"] = self.data["PD_DESC"].apply(
            lambda description: 3 if len(description) > 40 else
            2 if len(description) > 30 else
            1 if len(description) > 20 else
            0
        )

    def _encode_OFNS_DESC(self):
        """
        Encodes the OFNS_DESC column using Label Encoding, assigning each unique
        offense description a unique integer. NaN values are replaced with 'UNKNOWN'.
        """
        self.data['OFNS_DESC'] = self.data['OFNS_DESC'].fillna('UNKNOWN')  # Replace NaN with 'UNKNOWN'
        label_encoder = LabelEncoder()
        self.data['OFNS_DESC'] = label_encoder.fit_transform(self.data['OFNS_DESC'])

    def _encode_LAW_CODE(self):
        """Encode the LAW_CODE variable using Label Encoding"""
        self.data['LAW_CODE'] = self.data['LAW_CODE'].fillna('UNKNOWN')  # Replace NaN with 'UNKNOWN'
        label_encoder = LabelEncoder()
        self.data['LAW_CODE'] = label_encoder.fit_transform(self.data['LAW_CODE'])

    def _encode_LAW_CAT_CD(self):
        # Encode the law category code variable
        self.data['LAW_CAT_CD'] = self.data['LAW_CAT_CD'].map({'F': 1, 'M': 0})

    def _encode_ARREST_BORO(self):
        # Encode the borough variable
        self.data['ARREST_BORO'] = self.data['ARREST_BORO'].map({'M': 1, 'B': 2, 'Q': 3, 'K': 4, 'S': 5})

    def _encode_AGE_GROUP(self):

        age_group_mapping = {'UNKNOWN': 0, '<18': 1, '18-24': 2, '25-44': 3, '45-64': 4, '65+': 5}
        self.data['AGE_GROUP'] = self.data['AGE_GROUP'].map(age_group_mapping)

    def _encode_PERP_SEX(self):
        # Encode the perpetrator's sex variable
        self.data['PERP_SEX'] = self.data['PERP_SEX'].map({'M': 1, 'F': 0})

    def _encode_PERP_RACE(self):
        # Encode the perpetrator's race variable
        self.data['PERP_RACE'] = self.data['PERP_RACE'].map({'UNKNOWN': 0, 'BLACK': 1, 'WHITE HISPANIC': 2,
                                                             'WHITE': 3, 'BLACK HISPANIC': 4,
                                                             'ASIAN / PACIFIC ISLANDER': 5,
                                                             'AMERICAN INDIAN/ALASKAN NATIVE': 6, 'OTHER': 7})

    def encode_NY_ARRESTS(self):
        # Encode the simbolic values of the NY arrests dataset

        self._encode_ARREST_DATE()
        self._encode_PD_DESC()
        self._encode_OFNS_DESC()
        self._encode_LAW_CODE()
        self._encode_LAW_CAT_CD()
        self._encode_ARREST_BORO()
        self._encode_AGE_GROUP()
        self._encode_PERP_SEX()
        self._encode_PERP_RACE()

        print("Data encoding completed for the NY Arrests dataset.")