import pandas as pd

from classes.DataLoader import DataManipulator
from classes.DataProfiling import DataVisualization

# Load the data
path_dataset1_class = "data/class_ny_arrests.csv"
path_dataset2_class = "data/class_financial distress.csv"

data_loader1 = DataManipulator(path_dataset1_class, "LAW_CAT_CD")
data_loader2 = DataManipulator(path_dataset2_class, "CLASS")

# Display the data
print(data_loader1.data.head())
print(data_loader2.data.head())

# Data Encoding

# Data Visualization
data_visualization1 = DataVisualization(data_loader1)
data_visualization2 = DataVisualization(data_loader2)

data_visualization1.plot_basic_statistics()
data_visualization2.plot_basic_statistics()

data_visualization1.plot_variable_types()
data_visualization2.plot_variable_types()

data_visualization1.plot_missing_values()
data_visualization2.plot_missing_values()

data_visualization1.plot_global_boxplots()
data_visualization2.plot_global_boxplots()