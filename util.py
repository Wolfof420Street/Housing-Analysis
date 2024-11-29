import pandas as pd
import numpy as np

def load_dataset(filepath):
    """
    Load and initial preprocessing of dataset
    """
    data = pd.read_csv(filepath)
    
    # Basic cleaning
    data.dropna(subset=['SalePrice'], inplace=True)
    
    # Create total area feature
    data['TotalArea'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    # Encode categorical variables
    categorical_columns = ['Neighborhood', 'SaleType', 'SaleCondition']
    data = pd.get_dummies(data, columns=categorical_columns)
    
    return data

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]