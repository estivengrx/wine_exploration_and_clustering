# machine_learning_script.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score, precision_score, recall_score, f1_score

def preprocess_data(data):
    # Data scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled

def hyperparameter_tuning(data_scaled):
    # Parameters that will be tuned
    n_init_range = list(range(1, 10))
    param_grid = {'n_init': n_init_range,
                  'init': ['k-means++', 'random'],
                  'max_iter': [100, 200, 300, 400, 500, 600]}

    # Search for the best parameters with GridSearchCV from sklearn
    grid = GridSearchCV(KMeans(), param_grid, cv=10, scoring='accuracy', refit='silhouette_score')
    grid.fit(data_scaled)

    # Best parameters selected in the grid search
    print(grid.best_params_)

def train_kmeans(data_scaled, optimal_clusters=3):
    # Training the KMeans model with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=100, n_init=1, random_state=42)
    pred_y = kmeans.fit_predict(data_scaled)

    # Predicted clusters
    data_transformed['Cluster'] = pred_y

    # Calculating the silhouette score for the KMeans model
    score = silhouette_score(data_scaled, pred_y)
    inertia_score = kmeans.inertia_
    print(f"Silhouette score: {round(score, 2)}")
    print(f"Inertia score (WCSS): {round(inertia_score, 2)}")

    # Pairplot to visualize the relationships between the variables and the clusters
    sns.set(font_scale=1.5)
    plt.figure(figsize=(15, 15))

    palette = sns.color_palette("dark", len(data_transformed['Cluster'].unique()))
    g = sns.pairplot(data_transformed, hue='Cluster', diag_kind='kde', plot_kws={'alpha': 0.6}, palette=palette)
    g.fig.suptitle("Pairplot of Transformed Data", y=1.02, fontsize=20)

    # Making the legend bigger
    plt.setp(g._legend.get_title(), fontsize='25')
    plt.setp(g._legend.get_texts(), fontsize='23')

    plt.show()

    # Other analysis and visualization...

if __name__ == "__main__":
    # Load the preprocessed data (output from the data exploration script)
    data_transformed = pd.read_csv("path_to_preprocessed_data.csv")  # Provide the correct path

    # Perform data preprocessing
    data_scaled = preprocess_data(data_transformed.drop(columns=['Cluster']))

    # Hyperparameter tuning for KMeans
    hyperparameter_tuning(data_scaled)

    # Train the KMeans model
    train_kmeans(data_scaled)
