import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

class HousePricePrediction:
    def __init__(self, data_path):
        # Load data
        self.data = pd.read_csv(data_path)
        
        # Preprocessing variables
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Model variables
        self.best_model = None
        self.model_results = None
    
    def preprocess_data(self):
        # Print dataset info for debugging
        print("Dataset Shape:", self.data.shape)
        print("Columns:", self.data.columns.tolist())
        
        # Ensure dataset is not empty
        if self.data.empty:
            raise ValueError("Dataset is empty. Check your CSV file.")
        
        # Handle missing values
        self.data.dropna(subset=['SalePrice'], inplace=True)
        
        # Ensure required features exist
        required_columns = [
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
            'OverallQual', 'YearBuilt', 
            'TotRmsAbvGrd', 'GarageArea', 'FullBath'
        ]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Feature engineering
        self.data['TotalArea'] = (
            self.data['TotalBsmtSF'] + 
            self.data['1stFlrSF'] + 
            self.data['2ndFlrSF']
        )
        
        # Select features
        features = [
            'TotalArea', 'OverallQual', 'YearBuilt', 
            'TotRmsAbvGrd', 'GarageArea', 'FullBath'
        ]
        self.X = self.data[features]
        self.y = self.data['SalePrice']
        
        # Check for valid data before splitting
        if self.X.empty:
            raise ValueError("No data available after preprocessing.")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def compare_models(self):
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
            
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            results[name] = -scores.mean()
        
        self.model_results = results
        self.best_model = min(results, key=results.get)
        
        print("Model Comparison Results:")
        for model, score in results.items():
            print(f"{model}: MSE = {score:,.2f}")
        print(f"\nBest Model: {self.best_model}")
    
    def train_best_model(self):
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', models[self.best_model])
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        self.best_model = pipeline
    
    def evaluate_model(self):
        y_pred = self.best_model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\nModel Evaluation:")
        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"Mean Absolute Error: {mae:,.2f}")
        print(f"R-squared Score: {r2:.4f}")
    
    def predict_price(self, features):
        # Ensure features match training data columns
        return self.best_model.predict([features])[0]
    
    def visualize_results(self):
        plt.figure(figsize=(10, 6))
        plt.bar(self.model_results.keys(), self.model_results.values())
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Update with your dataset path
    predictor = HousePricePrediction('train.csv')
    
    # Run project pipeline
    predictor.preprocess_data()
    predictor.compare_models()
    predictor.train_best_model()
    predictor.evaluate_model()
    predictor.visualize_results()
    
    # Example prediction
    sample_house = [2000, 7, 2010, 8, 500, 2]  # Match feature order
    predicted_price = predictor.predict_price(sample_house)
    print(f"\nPredicted House Price: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
