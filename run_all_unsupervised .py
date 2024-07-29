import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt

## Function to Get Model and Parameter Grid

def un_get_model_and_param_grid(model_type):
    if model_type == "KMeans":
        model = KMeans()
        apply_scaling = True
        param_grid = {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'init': ['k-means++', 'random'],
            'max_iter': [300, 600, 900]
        }
    elif model_type == "AgglomerativeClustering":
        model = AgglomerativeClustering()
        apply_scaling = True
        param_grid = {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'linkage': ['ward', 'complete', 'average', 'single']
        }
    elif model_type == "DBSCAN":
        model = DBSCAN()
        apply_scaling = True
        param_grid = {
            'eps': [0.3, 0.5, 0.7, 1.0],
            'min_samples': [5, 10, 15, 20]
        }
    elif model_type == "PCA":
        model = PCA()
        apply_scaling = True
        param_grid = {
            'n_components': [2, 3, 4, 5]
        }
    else:
        raise ValueError("Invalid model type provided.")
    
    return model, param_grid, apply_scaling

## Function to Evaluate Clustering Models
def evaluate_clustering_model(df, feature_list, model_type):
    X = df[feature_list]
    
    try:
        # Initialize the model and parameter grid
        model, param_grid, apply_scaling = get_model_and_param_grid(model_type)

        # Apply scaling if necessary
        if apply_scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Perform GridSearchCV for clustering models
        best_model = None
        best_score = -1
        
        for params in [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]:
            model.set_params(**params)
            if model_type in ["KMeans", "AgglomerativeClustering"]:
                cluster_labels = model.fit_predict(X)
            elif model_type == "DBSCAN":
                model.fit(X)
                cluster_labels = model.labels_
            elif model_type == "PCA":
                transformed_data = model.fit_transform(X)
                plt.figure(figsize=(10, 7))
                sns.scatterplot(x=transformed_data[:, 0], y=transformed_data[:, 1])
                plt.title(f'PCA - {params}')
                plt.show()
                continue
            
            score = silhouette_score(X, cluster_labels)
            if score > best_score:
                best_score = score
                best_model = model

        # Use the best model to predict clusters
        if model_type in ["KMeans", "AgglomerativeClustering"]:
            best_model.set_params(**best_model.get_params())
            cluster_labels = best_model.fit_predict(X)
        elif model_type == "DBSCAN":
            best_model.fit(X)
            cluster_labels = best_model.labels_

        # Silhouette Score
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        print(f"Model: {model_type}")
        print("Best Parameters:", best_model.get_params())
        print("Silhouette Score:", silhouette_avg)

        # Plot clusters
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cluster_labels, palette='viridis')
        plt.title(f'{model_type} Clustering - Best Params: {best_model.get_params()}')
        plt.show()
        
        # Return metrics
        return {
            'model_type': model_type,
            'best_params': best_model.get_params(),
            'silhouette_score': silhouette_avg
        }
    except ValueError as e:
        print(e)
        return None

 ## Function to Evaluate Multiple Clustering Models
def evaluate_clustering_models(models, df, feature_list):
    results = []

    for model in models:
        result = evaluate_clustering_model(df, feature_list, model)
        if result is not None:
            results.append(result)

    # Convert results to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(results)

    # Plot the metrics comparison for all models
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model_type', y='silhouette_score', data=metrics_df)
    plt.title('Silhouette Score Comparison')
    plt.ylabel('Silhouette Score')
    plt.xlabel('Model Type')
    plt.show()

## Usage Example
'''
import itertools

# Example usage
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'feature4': np.random.rand(100)
})

# Define the feature list
feature_list = ['feature1', 'feature2', 'feature3', 'feature4']

# List of models to evaluate
models = ['KMeans', 'AgglomerativeClustering', 'DBSCAN', 'PCA']

# Evaluate the clustering models
evaluate_clustering_models(models, df, feature_list)

'''