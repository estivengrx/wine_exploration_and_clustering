# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uvicorn import run
from fastapi import FastAPI
from requests import get
from io import StringIO
from warnings import filterwarnings
from scipy.stats import kurtosis, skew
from ydata_profiling import ProfileReport
from matplotlib import patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from contextlib import redirect_stdout

# Initialize the Flask application
app = FastAPI()

@app.get('/')
def get_data() -> str:
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv' # Direct link to the dataset
    response = get(url)

    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text)) # Dataframe from the csv url
        print('Data retrieved successfully')
    else:
        print(f'Failed to download dataset. Status code: {response.status_code}')
    return data.to_json() # Return the data as a JSON object

# FastAPI route to retrieve and explore data
@app.get("/data-exploration")
def explore_data():
    # Retrieve data from the API
    data = get_data()
    data = pd.read_json(data)  # Convert the JSON object to a DataFrame

    # Perform data exploration and analysis
    result = data_exploration_and_analysis(data)

    return result

def calculate_stats(data: pd.DataFrame) -> dict:
    mean, std, skewness, kurtosis = [], [], [], []

    for column in data.columns:
        mean.append(data[column].mean())
        std.append(data[column].std())
        skewness.append(skew(data[column]))
        kurtosis.append(kurtosis(data[column]))

    stats_dict = {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

    return stats_dict

def remove_outliers(data):
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    data_transformed = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data_transformed

def visualize_boxplots(data, data_transformed, columns_affected):
    sns.set(font_scale=1.8)
    total_cols = 3
    num_plots = len(data[columns_affected].columns)
    total_rows = num_plots // total_cols + 1
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                            figsize=(7 * total_cols, 7 * total_rows), constrained_layout=True)

    flierprops_orig = dict(marker='o', markerfacecolor='red', markersize=6,
                           linestyle='none', markeredgecolor='red')
    flierprops_transformed = dict(marker='o', markerfacecolor='blue', markersize=6,
                                  linestyle='none', markeredgecolor='blue')

    for i, var in enumerate(data[columns_affected].columns):
        row = i // total_cols
        pos = i % total_cols
        plot = sns.boxplot(x=var, data=data[columns_affected], ax=axs[row][pos], flierprops=flierprops_orig,
                           color='lightblue')
        plot = sns.boxplot(x=var, data=data_transformed[columns_affected], ax=axs[row][pos],
                           flierprops=flierprops_transformed, color='#78ACA5')
        plot.set_title(f'{var}', fontdict={'fontsize': 20})
        plot.grid(alpha=0.5)

        # Add legend
        red_patch = mpatches.Patch(color='red', label='Outliers eliminated')
        lightblue_patch = mpatches.Patch(color='lightblue', label='Original Data')
        otherblue_patch = mpatches.Patch(color='#78ACA5', label='Transformed Data')
        plot.legend(handles=[red_patch, lightblue_patch, otherblue_patch], loc='upper right')

    # Remove empty subplots
    if num_plots % total_cols:
        for ax in axs.flatten()[num_plots:]:
            ax.remove()

    plt.suptitle('Boxplots for each variable affected by the outlier treatment', fontsize=30)
    plt.show()

def generate_profiling_report(data_transformed):
    profile_report = ProfileReport(data_transformed, title='Wine clustering profiling report',
                                   explorative=True, correlations={"spearman": {"calculate": True}})
    profile_report.to_notebook_iframe()

    matrix_correlations_from_report = profile_report._description_set.correlations['spearman']
    mask = np.abs(matrix_correlations_from_report.values) > 0.8
    relationships = [(matrix_correlations_from_report.columns[i],
                      matrix_correlations_from_report.columns[j]) \
                     for i, j in zip(*np.where(mask)) if i != j]

    relationships_tuples = [tuple(sorted(rel)) for rel in relationships]
    unique_relationships = list(set(relationships_tuples))

    print(f'Relationships with a coefficient greater than 0.7: {unique_relationships}')

def data_exploration_and_analysis(data):
    # Capture print output
    output = StringIO()
    filterwarnings('ignore')

    # Redirect print statements to StringIO
    with redirect_stdout(output):
        filterwarnings('ignore')
        dimensions = str(data.shape)
        null_data = str(data.isnull().sum().sum())
        duplicated_data = str(data.duplicated().sum())
        inf_rows = str((data == np.inf).sum(axis=1).sum())
        inf_columns = str((data == np.inf).sum().sum())
        data_info = str(data.info())
        data_description = str(data.describe())

        stats_data = {
            'original_data': calculate_stats(data),
            'transformed_data': calculate_stats(remove_outliers(data))
        }

        rows_deleted = len(data) - len(remove_outliers(data))

        sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
        plt.suptitle("Pairplot of the data", y=1.02, size=20)
        plt.show()

        columns_affected = ['Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Proanthocyanins', 'Color_Intensity', 'Hue']
        visualize_boxplots(data, remove_outliers(data), columns_affected)

        generate_profiling_report(remove_outliers(data))

    result = output.getvalue()

    exploration_results = {
        "dimensions": dimensions,
        "null_data": null_data,
        "duplicated_data": duplicated_data,
        "inf_rows": inf_rows,
        "inf_columns": inf_columns,
        "data_info": data_info,
        "data_description": data_description,
        "stats_data": stats_data,
        "rows_deleted": rows_deleted,
        "visualization_result": result
    }

    return exploration_results

@app.get("/clustering-analysis")
def perform_clustering_analysis():
    # Retrieve data from the API
    data = get_data()
    data = pd.read_json(data)

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

# Run the Flask application
if __name__ == '__main__':
    run(app, port=8000, host='127.0.0.1', log_level='info')