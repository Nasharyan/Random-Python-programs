# Import necessary modules and functions
from preprocessing_of_data import preprocess_data
import Evaluation
from sklearn.model_selection import train_test_split
import pandas as pd

# Preprocess the data
X, y = preprocess_data('Asteroid.csv')

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the model
best_model, model_results = Evaluation.evaluate_model(X_train, y_train, X_val, y_val)

# Print model evaluation results
print("Model Evaluation Results:")
print(model_results)
