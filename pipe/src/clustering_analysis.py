# clustering_analysis_script.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_extraction_api
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from fastapi import FastAPI
from io import StringIO
from contextlib import redirect_stdout

app = FastAPI()
@app.get("/clustering-analysis")
def perform_clustering_analysis():
    # Retrieve data from the API
    data = retrieve_data()

    # Perform data preprocessing
    data_scaled, data_transformed = preprocess_data(data)

    # Hyperparameter tuning
    best_params = tune_hyperparameters(data_scaled)

    # Model training
    pred_y, score, inertia_score = train_model(data_scaled, best_params)

    # Data visualization
    cluster_means = visualize_data(data_transformed, pred_y)

    # Create a dictionary with all the results
    clustering_results = {
        "best_params": best_params,
        "silhouette_score": round(score, 2),
        "inertia_score": round(inertia_score, 2),
        "cluster_means": cluster_means.to_dict()
    }

    return clustering_results

def retrieve_data():
    data = data_extraction_api.get_data()
    data = pd.read_json(data)  # Convert the JSON object to a DataFrame
    return data

def preprocess_data(data):
    # Interquartile Range (IQR) method to remove outliers
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    data_transformed = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Data scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_transformed)

    return data_scaled, data_transformed

def tune_hyperparameters(data_scaled):
    n_init_range = list(range(1, 10))
    param_grid = {'n_init': n_init_range,
                  'init': ['k-means++', 'random'],
                  'max_iter': [100, 200, 300, 400, 500, 600]}

    grid = GridSearchCV(KMeans(), param_grid, cv=10, scoring='accuracy', refit='silhouette_score')
    grid.fit(data_scaled)

    # Best parameters selected in the grid search
    best_params = grid.best_params_

    return best_params

def train_model(data_scaled, best_params):
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=100, n_init=1, random_state=42)
    pred_y = kmeans.fit_predict(data_scaled)

    # Calculating silhouette score for KMeans model
    score = silhouette_score(data_scaled, pred_y)
    inertia_score = kmeans.inertia_

    return pred_y, score, inertia_score

def visualize_data(data_transformed, pred_y):
    # -- Data Grouping and Visualization --
    data_transformed['Cluster'] = pred_y
    cluster_means = data_transformed.groupby('Cluster').mean()

    # -- Plotting Pairplot --
    sns.set(font_scale=1.5)
    plt.figure(figsize=(15, 15))
    palette = sns.color_palette("dark", len(data_transformed['Cluster'].unique()))
    g = sns.pairplot(data_transformed, hue='Cluster', diag_kind='kde', plot_kws={'alpha': 0.6}, palette=palette)
    g.fig.suptitle("Pairplot of Transformed Data", y=1.02, fontsize=20)

    # Making the legend bigger
    plt.setp(g._legend.get_title(), fontsize='25')
    plt.setp(g._legend.get_texts(), fontsize='23')

    plt.show()

    return cluster_means

# Entry point of the script
if __name__ == "__main__":
    import uvicorn

    # Run FastAPI application using uvicorn
    uvicorn.run(app, port=8000, host='127.0.0.1', log_level="info")