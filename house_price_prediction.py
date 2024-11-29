import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression
import statsmodels.api as sm
from joblib import dump


class HousePricePrediction:
    def __init__(self, data_path):
        """
        Initialize the house price prediction project
        
        Theoretical Concept:
        - Data-driven approach to predict house prices
        - Leveraging machine learning regression techniques
        """
        self.data = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
    
    def exploratory_data_analysis(self):
        # Select numeric columns with strong correlation to SalePrice
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        correlation_with_price = self.data[numeric_columns].corr()['SalePrice'].abs()
        top_features = correlation_with_price[correlation_with_price > 0.5].index.tolist()
        
        # Correlation Heatmap of Top Features
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data[top_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                    linewidths=0.5, fmt=".2f", square=True, cbar_kws={"shrink": .8})
        plt.title('Top Features Correlation Heatmap')
        plt.tight_layout()
        plt.show()


    def feature_engineering(self):
        """
        Advanced Feature Engineering
        
        Theoretical Concept:
        - Create derived features to capture complex relationships
        - Enhance predictive power through feature transformation
        """
        # Derive new features
        self.data['TotalArea'] = (
            self.data['TotalBsmtSF'] + 
            self.data['1stFlrSF'] + 
            self.data['2ndFlrSF']
        )
        
        # Age-related feature
        self.data['HouseAge'] = 2023 - self.data['YearBuilt']
        
        # Price per square foot
        self.data['PricePerSqFt'] = self.data['SalePrice'] / self.data['TotalArea']
        
        # Select and prepare features
        features = [
            'TotalArea', 'OverallQual', 'YearBuilt', 
            'TotRmsAbvGrd', 'GarageArea', 'FullBath', 
            'HouseAge', 'PricePerSqFt'
        ]
        
        self.X = self.data[features]
        self.y = self.data['SalePrice']
        
        # Variance Inflation Factor (VIF) Analysis
        X_with_const = sm.add_constant(self.X)
        vif = pd.DataFrame()
        vif["Variable"] = X_with_const.columns
        vif["VIF"] = [sm.OLS(X_with_const[col], X_with_const.drop(columns=[col])).fit().rsquared_adj for col in X_with_const.columns]
        
        print("\nVariance Inflation Factor Analysis:")
        print(vif)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def model_comparison(self):
        """
        Advanced Model Comparison
        
        Theoretical Concept:
        - Evaluate multiple regression algorithms
        - Compare performance using cross-validation
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
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
        
        # Visualize model comparison
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Select best model
        self.best_model_name = min(results, key=results.get)
        print(f"\nBest Model: {self.best_model_name}")
    
    def train_and_evaluate(self):
        """
        Model Training and Comprehensive Evaluation
        
        Theoretical Concept:
        - Train selected model
        - Provide detailed performance metrics
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', models[self.best_model_name])
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        self.best_model = pipeline

        # Save the trained model to a file
        dump(self.best_model, 'trained_model.joblib')
        print("\nTrained model saved as 'trained_model.joblib'.")
        
        # Predictions
        y_pred = self.best_model.predict(self.X_test)
        
        # Evaluation Metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\nModel Evaluation:")
        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"Mean Absolute Error: {mae:,.2f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Residual Analysis
        residuals = self.y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()

    def predict_price(self, features):
        """
        Price Prediction Method
        
        Theoretical Concept:
        - Transform input features
        - Utilize trained model for prediction
        """
        return self.best_model.predict([features])[0]

def main():
    # Project Workflow
    predictor = HousePricePrediction('train.csv')
    
    # Weekly Project Steps
    predictor.exploratory_data_analysis()
    predictor.feature_engineering()
    predictor.model_comparison()
    predictor.train_and_evaluate()
    
    # Example Prediction
    sample_house = [2000, 7, 2010, 8, 500, 2, 13, 150]
    predicted_price = predictor.predict_price(sample_house)
    print(f"\nPredicted House Price: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()