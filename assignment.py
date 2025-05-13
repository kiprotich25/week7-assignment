import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First 5 rows of the dataset:\n", df.head())
    print("\nData types and missing values:\n", df.info())


    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print("Missing values dropped.")
    else:
        print("No missing values found.")

except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2
print("\nDescriptive statistics:\n", df.describe())

# Group by species and calculate mean
grouped = df.groupby('species').mean()
print("\nMean values per species:\n", grouped)

# Task 3: Data Visualization

# 1. Line Chart - Simulate trend over index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Sepal and Petal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar Chart - Average petal length by species
plt.figure(figsize=(7, 5))
species = grouped.index
avg_petal_length = grouped['petal length (cm)']
plt.bar(species, avg_petal_length, color='mediumseagreen')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram - Sepal Width Distribution
plt.figure(figsize=(7, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(7, 5))
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'],
                label=species, color=colors[species])
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

