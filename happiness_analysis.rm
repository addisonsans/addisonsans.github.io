# Method 1: Direct data loading in the notebook
import pandas as pd
from io import StringIO

# Your data as a CSV string
data_str = '''Country name,Iso alpha,year,Happiness score,Log GDP per capita,Social support,Healthy life expectancy at birth,Freedom to make life choices,Generosity,Perceptions of corruption,Positive affect,Negative affect
Afghanistan,AFG,2008,3.724,7.35,0.451,50.5,0.718,0.168,0.882,0.414,0.258
Afghanistan,AFG,2009,4.402,7.509,0.552,50.8,0.679,0.191,0.85,0.481,0.237
# ... paste all your data rows here ...'''

# Load the data
df = pd.read_csv(StringIO(data_str))

# Clean column names
df.columns = ['country', 'iso', 'year', 'happiness', 'gdp', 'social', 
              'health', 'freedom', 'generosity', 'corruption', 
              'positive_affect', 'negative_affect']

# Method 2: Load from a separate file
# Save your data as 'happiness_data.csv' in the same directory as your notebook
# Then use:
# df = pd.read_csv('happiness_data.csv')

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
