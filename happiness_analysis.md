# Analyzing Global Happiness Trends: A Data Science Approach

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/happiness_analysis.ipynb)

## Introduction
In this analysis, we'll explore the World Happiness Report data to uncover trends and relationships between happiness and various socioeconomic factors. We'll use Python and popular data science libraries to perform our analysis.

## Setup and Data Import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Read the data
def load_happiness_data():
    # Create a StringIO object with the data
    data = pd.read_csv(StringIO("""Country name,Iso alpha,year,Happiness score,Log GDP per capita,Social support,Healthy life expectancy at birth,Freedom to make life choices,Generosity,Perceptions of corruption,Positive affect,Negative affect
    ...[your data here]..."""))
    
    # Rename columns for easier handling
    data.columns = ['country', 'iso', 'year', 'happiness', 'gdp', 'social', 
                   'health', 'freedom', 'generosity', 'corruption', 
                   'positive_affect', 'negative_affect']
    return data

# Load the data
df = load_happiness_data()

# Display basic information about the dataset
print("Dataset Overview:")
print(f"Number of countries: {df['country'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print("\nMissing values:")
print(df.isnull().sum())
```

## Exploratory Data Analysis

### Happiness Score Distribution

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='happiness', bins=30)
plt.title('Distribution of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Count')
plt.show()

print("\nHappiness Score Statistics:")
print(df['happiness'].describe())
```

### Correlation Analysis

```python
# Select numeric columns for correlation analysis
numeric_cols = ['happiness', 'gdp', 'social', 'health', 'freedom', 
                'generosity', 'corruption']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Happiness Factors')
plt.tight_layout()
plt.show()
```

### Time Series Analysis

```python
# Calculate average happiness by year
yearly_happiness = df.groupby('year')['happiness'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(yearly_happiness['year'], yearly_happiness['happiness'], 
         marker='o', linewidth=2)
plt.title('Global Average Happiness Score Over Time')
plt.xlabel('Year')
plt.ylabel('Average Happiness Score')
plt.grid(True)
plt.show()
```

### Regional Analysis

```python
# Get most recent year's data
latest_year = df['year'].max()
latest_data = df[df['year'] == latest_year]

# Top 10 happiest countries
print("\nTop 10 Happiest Countries ({}):".format(latest_year))
print(latest_data.nlargest(10, 'happiness')[['country', 'happiness']])

# Bottom 10 happiest countries
print("\nBottom 10 Happiest Countries ({}):".format(latest_year))
print(latest_data.nsmallest(10, 'happiness')[['country', 'happiness']])
```

### GDP vs Happiness Analysis

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=latest_data, x='gdp', y='happiness')
plt.title('GDP vs Happiness Score')
plt.xlabel('Log GDP per capita')
plt.ylabel('Happiness Score')

# Add regression line
x = latest_data['gdp']
y = latest_data['happiness']
z = np.polyfit(x[~np.isnan(x) & ~np.isnan(y)], 
               y[~np.isnan(x) & ~np.isnan(y)], 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8)

# Calculate correlation
correlation = latest_data['gdp'].corr(latest_data['happiness'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
         transform=plt.gca().transAxes)

plt.show()
```

## Statistical Analysis

### Multiple Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Prepare data for regression
X = latest_data[['gdp', 'social', 'health', 'freedom', 'generosity']]
y = latest_data['happiness']

# Remove rows with missing values
mask = ~X.isna().any(axis=1) & ~y.isna()
X = X[mask]
y = y[mask]

# Fit regression model
model = LinearRegression()
model.fit(X, y)

# Print results
print("\nMultiple Linear Regression Results:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.3f}")
print(f"R-squared: {r2_score(y, model.predict(X)):.3f}")
```

## Key Findings

1. GDP per capita shows the strongest correlation with happiness (correlation coefficient: {correlation:.2f})
2. Social support and health expectancy are also strongly correlated with happiness
3. The global average happiness score has [increased/decreased] over the study period
4. Nordic countries consistently rank among the happiest nations
5. Multiple regression analysis suggests that [X]% of happiness score variance can be explained by the measured factors

## Conclusions and Future Work

This analysis reveals strong relationships between economic factors, social support, and happiness levels. Future work could explore:
- Causality analysis using time-series methods
- Regional and cultural factors affecting happiness
- Impact of specific policy changes on happiness scores

## How to Reproduce This Analysis

1. Click the "Open in Colab" button at the top of this notebook
2. Run all cells in sequence
3. Modify parameters or analysis methods as needed

---

*Note: This analysis uses data from the World Happiness Report. All code is available in the accompanying GitHub repository.*
