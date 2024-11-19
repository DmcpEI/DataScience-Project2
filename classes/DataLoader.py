import pandas as pd

class DataLoader:
    """
    Generic Class responsible for loading the dataset

    Parameters:
        filename (str): The filename of the dataset to load.
        target (str): The target of the dataset to load.

    Attributes (after loading the data):
        data (DataFrame): The main dataset containing both features and target variable.
        labels (DataFrame): The target variable.
        numerical_features (List): List of numerical features in the dataset.
        categorical_features (List): List of categorical features in the dataset.


    Methods:
        _load_data(): Loads the dataset,and assigns the data and labels to the appropriate attributes.
    """

    def __init__(self, filename, target):
        """
        Initializes the DataLoader with the filename of the dataset.

        Parameters:
            filename (str): The filename of the dataset to load.
            target (str): The target of the dataset to load.
        """
        self.filename = filename
        self.file_tag = filename.split("/")[-1].split(".")[0]

        self.data = None
        self.target = target
        self.labels = None
        self.numerical_features = []
        self.categorical_features = []

        # Load data
        self._load_data(target)

    def _load_data(self, target):
        """
        Loads the dataset from the specified filename,
        and assigns the data and labels to the appropriate attributes.

        Parameters:
            target (str): The target of the dataset to load.
        """
        try:
            # Load the dataset
            self.data = pd.read_csv(self.filename)

            # Validate if the target column exists in the dataset
            if target not in self.data.columns:
                raise ValueError(f"Target column '{target}' not found in the dataset.")

            self.labels = self.data[target]

            print("Data loaded successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")

class DataManipulator(DataLoader):
    """
    A class for manipulating data loaded from a file.

    Parameters:
        filename (str): The path to the data file.
        target (str): The target variable in the data.

    Attributes:
        data (DataFrame): The loaded data.

    Methods:
        _describe_variables: Prints information about the data, including data info, unique values, and statistical distribution.

    Raises:
        FileNotFoundError: If the specified file is not found.

    """

    def __init__(self, filename, target):
        """
        Initialize the class with a filename and target variable.

        Parameters:
            filename (str): The path to the file.
            target (str): The name of the target variable.

        Raises:
            FileNotFoundError: If the file is not found.

        """
        try:
            super().__init__(filename, target)
            print("\nData Description:")
            self._describe_variables()
        except FileNotFoundError:
            print("File not found. Please check the file path.")

    def _describe_variables(self):
        """
        Prints information about the data, including data info, unique values, and statistical distribution.
        """
        print("\nInformation of Data:")
        print(self.data.info())

        print("\nUnique values of features:")
        print(self.data.nunique())

        print("\nStatistical distribution of each variable:")
        print(self.data.describe())

    def update_data(self, filename):
        """
        Updates the data attribute with the data from the specified file.

        Parameters:
            filename (str): The path to the file.

        Raises:
            FileNotFoundError: If the file is not found.

        """
        try:
            self.data = pd.read_csv(filename)
            print("Data updated successfully.")
        except FileNotFoundError:
            print("File not found. Please check the file path.")