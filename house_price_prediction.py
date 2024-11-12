import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(df.head())

# Descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize outliers using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Check data types
print(df.dtypes)

# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Example: Remove outliers beyond 3 standard deviations
df = df[(df - df.mean()).abs() <= 3 * df.std()]

# Normalize numerical features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the cleaned dataset
df.to_csv('cleaned_house_prices.csv', index=False)