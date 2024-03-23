import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# List of regression models to evaluate
models = [
    (LinearRegression(), 'Linear Regression'),
    (KNeighborsRegressor(n_neighbors=3), 'K-Nearest Neighbors'),
    (RandomForestRegressor(n_jobs=-1, max_samples=30000, random_state=42), 'Random Forest Regressor')
]

def evaluate_model(X_train, y_train, X_val, y_val):
    """
    Evaluate regression models and return the best model along with evaluation scores.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        X_val (DataFrame): Validation features.
        y_val (Series): Validation target.

    Returns:
        best_model (estimator): Best performing model.
        model_results (DataFrame): DataFrame containing evaluation scores for all models.
    """
    scores = []

    best_model = None
    best_score = -1  # Initialize best_score to a very low value

    for model, model_name in models:
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)
        
        # Store scores
        scores.append((model_name, mae, rmse, r2))
        
        # Update best model if current model has higher R2 score
        if r2 > best_score:
            best_score = r2
            best_model = model

    return best_model, pd.DataFrame(scores, columns=['Model', 'MAE', 'RMSE', 'R2 Score'])

def predict_with_best_model(best_model, X_val):
    """
    Generate predictions using the best model.

    Parameters:
        best_model (estimator): Best performing model.
        X_val (DataFrame): Validation features.

    Returns:
        predictions (array-like): Predicted target values.
    """
    if best_model in models:
        predictions = best_model.predict(X_val)
    return predictions
