import pandas as pd
from pandas import DataFrame


class DataLoader:
    """
    Generic Class responsible for loading the dataset.

    Parameters:
        filename (str): The filename of the dataset to load.
        target (str): The target of the dataset to load.
        read_options (dict): Additional options for reading the CSV file.

    Attributes:
        data (DataFrame): The main dataset containing both features and the target variable.
        labels (Series): The target variable.
    """

    def __init__(self, filename, target, read_options=None):
        """
        Initializes the DataLoader with the filename of the dataset.

        Parameters:
            filename (str): The filename of the dataset to load.
            target (str): The target of the dataset to load.
            read_options (dict): Additional options for reading the CSV file.
        """
        self.filename = filename
        self.file_tag = filename.split("/")[-1].split(".")[0]
        self.target = target

        self.data = None
        self.labels = None

        # Store read options for reading CSV
        self.read_options = read_options if read_options else {}

        # Load the data
        self._load_data()

    def _load_data(self):
        """
        Loads the dataset using the specified filename and options.
        """
        try:
            # Load the dataset with the provided options
            self.data: DataFrame = pd.read_csv(self.filename, **self.read_options)

            # Validate if the target column exists in the dataset
            if self.target not in self.data.columns:
                raise ValueError(f"Target column '{self.target}' not found in the dataset.")

            # Extract labels
            self.labels = self.data[self.target]
            print("Data loaded successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")
        except ValueError as ve:
            print(f"Error while loading data: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


class DataManipulator(DataLoader):
    """
    A class for manipulating data loaded from a file.

    Parameters:
        filename (str): The path to the data file.
        target (str): The target variable in the data.
        read_options (dict): Additional options for reading the CSV file.

    Methods:
        _describe_variables: Prints information about the data, including data info, unique values, and statistical distribution.
    """

    def __init__(self, filename, target, read_options=None):
        """
        Initialize the class with a filename and target variable.

        Parameters:
            filename (str): The path to the file.
            target (str): The name of the target variable.
            read_options (dict): Additional options for reading the CSV file.
        """
        super().__init__(filename, target, read_options)
        print("\nData Description:")
        self._describe_variables()

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

    def update_data(self, filename, read_options=None):
        """
        Updates the data attribute with the data from the specified file.

        Parameters:
            filename (str): The path to the file.
            read_options (dict): Additional options for reading the CSV file.
        """
        try:
            self.read_options = read_options if read_options else self.read_options
            self.data = pd.read_csv(filename, **self.read_options)
            print("Data updated successfully.")
        except FileNotFoundError:
            print("File not found. Please check the file path.")