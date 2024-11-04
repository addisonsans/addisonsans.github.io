# Method 1: Direct data loading in the notebook
import pandas as pd
from io import StringIO

# Clean column names
df.columns = ['country', 'iso', 'year', 'happiness', 'gdp', 'social', 
              'health', 'freedom', 'generosity', 'corruption', 
              'positive_affect', 'negative_affect']

# Method 2: Load from a separate file
# Save your data as 'happiness_data.csv' in the same directory as your notebook
# Then use:
# df = pd.read_csv('data/happiness_data.csv')

# Basic data check
print("Dataset Overview:")
print(f"Number of countries: {df['country'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())
