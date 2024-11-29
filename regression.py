from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def compare_regression_models(X, y):
    """
    Compare different regression models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }
    
    results = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        results[name] = -scores.mean()
    
    return results

def select_best_model(results):
    """
    Select the model with lowest mean squared error
    """
    return min(results, key=results.get)