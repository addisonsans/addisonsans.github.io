## Analyzing Correlates of Happiness 

The World Happiness Report provides rich data on various factors that potentially influence a country's overall happiness score. Let's examine a few of these and their relationship with the Happiness Score.

### GDP per Capita vs Happiness Score

One key variable is the log GDP per capita, a measure of a country's economic output and standard of living. We can look at how strongly this correlates with the Happiness Score.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('whr_200522.csv') 

plt.figure(figsize=(8,5))
plt.scatter(df['Log GDP per capita'], df['Happiness score'])
plt.xlabel('Log GDP per Capita')  
plt.ylabel('Happiness Score')
plt.title('GDP per Capita vs Happiness Score')
plt.show()
```

The scatter plot shows a clear positive correlation - countries with higher GDP per capita tend to have higher happiness scores. The relationship isn't perfectly linear, with some outliers, but the overall trend is evident. This suggests economic prosperity is an important, but not the sole, contributor to a nation's happiness.

### Social Support vs Happiness Score

Another interesting variable is Social Support, which captures the strength of social networks and relationships. Plotting this against the Happiness Score:

```python
plt.figure(figsize=(8,5))  
plt.scatter(df['Social support'], df['Happiness score'])
plt.xlabel('Social Support')
plt.ylabel('Happiness Score')
plt.title('Social Support vs Happiness Score')
plt.show()
```

Again, we see a positive correlation, indicating that countries where people feel they have strong social support tend to be happier overall. The relationship appears slightly stronger and more linear compared to GDP per capita.

### Healthy Life Expectancy vs Happiness Score

Lastly, let's look at Healthy Life Expectancy, a measure of population health and longevity.

```python
plt.figure(figsize=(8,5))
plt.scatter(df['Healthy life expectancy at birth'], df['Happiness score']) 
plt.xlabel('Healthy Life Expectancy')
plt.ylabel('Happiness Score')
plt.title('Healthy Life Expectancy vs Happiness Score') 
plt.show()
```

Once more, there's a clear positive relationship - countries with higher healthy life expectancy have higher happiness scores on average. Health appears to be another key determinant of national happiness.

## Conclusion

This brief analysis highlights how economic output, social connections, and population health all show positive correlations with overall happiness at a national level. However, happiness is a complex phenomenon and likely influenced by many other factors as well. The scatterplots also reveal outliers and variation around the trend lines, indicating that high GDP, social support or health don't automatically guarantee happiness.

Further analysis could examine other variables in the dataset, look at changes over time, or build predictive models of happiness. But this provides an interesting starting point in understanding some of the key correlates of happiness around the world. The World Happiness Report offers a wealth of data to explore this important question.

