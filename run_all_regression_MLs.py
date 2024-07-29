import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt

## Function to Get Model and Parameter Grid
def reg_get_model_and_param_grid(model_type):
    if model_type == "LinearRegression":
        model = LinearRegression()
        apply_scaling = False
        param_grid = {}
    elif model_type == "Ridge":
        model = Ridge()
        apply_scaling = True
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
    elif model_type == "Lasso":
        model = Lasso()
        apply_scaling = True
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
    elif model_type == "DecisionTree":
        model = DecisionTreeRegressor()
        apply_scaling = False
        param_grid = {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        }
    elif model_type == "RandomForest":
        model = RandomForestRegressor()
        apply_scaling = False
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['mse', 'mae'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        }
    elif model_type == "KNeighbors":
        model = KNeighborsRegressor()
        apply_scaling = True
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    elif model_type == "SVR":
        model = SVR()
        apply_scaling = True
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    else:
        raise ValueError("Invalid model type provided.")
    
    return model, param_grid, apply_scaling

## Function to Evaluate Regression Model
def evaluate_regression_model(df, target_var, feature_list, model_type):
    X = df[feature_list]
    y = df[target_var]
    
    try:
        # Initialize the model and parameter grid
        model, param_grid, apply_scaling = get_model_and_param_grid(model_type)

        # Apply scaling if necessary
        if apply_scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        # Get the best estimator
        optimized_clf = grid_search.best_estimator_
        
        # Predictions
        train_preds = optimized_clf.predict(X_train)
        test_preds = optimized_clf.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        print(f"Model: {model_type}")
        print("The MAE on train data is ", train_mae)
        print("The MAE on test data is ", test_mae)
        print("The MSE on train data is ", train_mse)
        print("The MSE on test data is ", test_mse)
        print("The R^2 on train data is ", train_r2)
        print("The R^2 on test data is ", test_r2)
        
        # Plot predicted vs actual values
        plt.figure(figsize=(10, 7))
        plt.scatter(y_test, test_preds, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted for {model_type}')
        plt.show()
        
        # Return metrics
        return {
            'model_type': model_type,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    except ValueError as e:
        print(e)
        return None

## Function to Evaluate Multiple Regression Models
def evaluate_regression_models(models, df, target_var, feature_list):
    results = []

    for model in models:
        result = evaluate_regression_model(df, target_var, feature_list, model)
        if result is not None:
            results.append(result)

    # Convert results to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(results)

    # Plot the metrics comparison for all models
    metrics = ['mae', 'mse', 'r2']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y=f'test_{metric}', data=metrics_df)
        plt.title(f'Test {metric.upper()} Comparison')
        plt.ylabel(f'Test {metric.upper()}')
        plt.xlabel('Model Type')
        plt.show()

## Usage Example
'''
# Define the target variable and feature list
target_var = 'target_column_name'
feature_list = ['feature1', 'feature2', 'feature3', 'feature4']

# List of models to evaluate
models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'KNeighbors', 'SVR']

# Evaluate the regression models
evaluate_regression_models(models, df, target_var, feature_list)

'''