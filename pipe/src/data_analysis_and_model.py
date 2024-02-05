import pandas as pd
import numpy as np
from uvicorn import run
from fastapi import FastAPI
from requests import get
from io import StringIO
from warnings import filterwarnings
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

# Initialize the Flask application
app = FastAPI()
filterwarnings('ignore')

## -- Data Extraction -- ##
# FastAPI route to retrieve data from a URL
@app.get('/')
def get_data() -> str:
    """
    Fetches data from a specified URL and returns it as a JSON string.

    The function sends a GET request to the URL of a CSV file. If the request is successful,
    it reads the CSV data into a pandas DataFrame, prints a success message, and returns the data
    as a JSON string.

    Returns:
        str: The data as a JSON string if the request is successful.
    """
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv' # Direct link to the dataset
    response = get(url)

    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text)) # Dataframe from the CSV URL
        print('Data retrieved successfully')
    else:
        print(f'Failed to download dataset. Status code: {response.status_code}')
    return data.to_json() # Return the data as a JSON object


## -- Data Exploration -- ##

def calculate_stats(data: pd.DataFrame) -> list:
    """
    Calculate the mean, standard deviation, skewness, and kurtosis for each column in a DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame for which to calculate statistics.

    Returns:
    mean (list): The mean of each column.
    std (list): The standard deviation of each column.
    skewness (list): The skewness of each column.
    kurtosi (list): The kurtosis of each column.
    """
    stats = []
    for column in data.columns:
        # Calculate your statistics here. For example:
        mean = data[column].mean()
        median = data[column].median()
        std_dev = data[column].std()
        kurtosis = data[column].kurtosis()
        skewness = data[column].skew()

        stats.append({
            'column': column,
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'kurtosis': kurtosis,
            'skewness': skewness
        })

    return stats

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the given DataFrame.

    This function uses the IQR method to identify and remove outliers. The IQR is the range between the first quartile (25 percentile)
    and the third quartile (75 percentile). Any data point that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier.

    Args:
        data (DataFrame): The data from which to remove outliers.

    Returns:
        DataFrame: The data with outliers removed.
    """

    # Calculate Q1 and Q3
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Identify and remove outliers
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    data_transformed = data[~outliers.any(axis=1)]

    return data_transformed

def greatest_relationships(data_transformed) -> list:
    """
    Identifies pairs of variables in the given data with a correlation coefficient greater than 0.7.

    This function creates a profiling report using the ydata_profiling library, which provides an in-depth analysis of the dataset.
    It then extracts the Spearman correlation matrix from the report and identifies pairs of variables with a correlation coefficient
    greater than 0.7. These pairs are returned as a list of tuples.

    Args:
        data_transformed (DataFrame): The transformed data for which to identify relationships.

    Returns:
        list: A list of tuples, where each tuple contains two variables that have a correlation coefficient greater than 0.7.
    """
    # Generate a profiling report for the transformed data
    profile_report = ProfileReport(data_transformed, title='Wine clustering profiling report',
                                   explorative=True, correlations={"spearman": {"calculate": True}})
    profile_report.to_notebook_iframe()

    # Extract the Spearman correlation matrix from the report
    matrix_correlations_from_report = profile_report._description_set.correlations['spearman']

    # Identify pairs of variables with a correlation coefficient greater than 0.7
    mask = np.abs(matrix_correlations_from_report.values) > 0.7
    relationships = [(matrix_correlations_from_report.columns[i],
                      matrix_correlations_from_report.columns[j]) \
                     for i, j in zip(*np.where(mask)) if i != j]

    # Sort the relationships and remove duplicates
    relationships_tuples = [tuple(sorted(rel)) for rel in relationships]
    unique_relationships = list(set(relationships_tuples))

    # Return the unique relationships
    return unique_relationships

def data_exploration_and_analysis(data: pd.DataFrame) -> dict:
    """
    Performs data exploration and analysis on the given DataFrame.

    This function calculates various statistics and information about the data, including dimensions, null data count,
    duplicated data count, infinite rows and columns count, descriptive statistics, outlier removal statistics,
    and relationships between variables.

    Args:
        data (DataFrame): The data to explore and analyze.

    Returns:
        dict: A dictionary containing the results of the data exploration and analysis.
    """

    # Calculate dimensions of the original dataset
    dimensions = str(data.shape)

    # Calculate total number of null data
    null_data = str(data.isnull().sum().sum())

    # Calculate total number of duplicated data
    duplicated_data = str(data.duplicated().sum())

    # Calculate total number of infinite rows and columns (can be considered null)
    inf_rows = str((data == np.inf).sum(axis=1).sum())
    inf_columns = str((data == np.inf).sum().sum())

    # Get descriptive statistics of the data
    data_description = str(data.describe())

    # Calculate total number of rows deleted due to outliers
    rows_deleted = len(data) - len(remove_outliers(data))

    # Calculate percentage of rows deleted due to outliers
    percentage_rows_deleted = (rows_deleted / len(data)) * 100

    # Calculate greatest relationships between variables (more than 0.7 in the spearman method)
    greatest_relationships_variables = greatest_relationships(remove_outliers(data))

    # Calculate statistics and distribution analysis to eliminate outliers
    stats_data = {
        'Statistics original data': calculate_stats(data),
        'Statistics data without outliers': calculate_stats(remove_outliers(data))
    }

    # Organize results in a dictionary
    exploration_results = {
        "Dimensions of the orginal dataset": dimensions,
        "Total number of null data": null_data,
        "Total number of duplicated data": duplicated_data,
        "Total number of infinite rows, (can be considered null)": inf_rows,
        "Total number of infinite columns, (can be considered null)": inf_columns,
        "Total number of rows deleted due to outliers": rows_deleted,
        "Percenatge of rows deleted due to outliers": percentage_rows_deleted,
        "Greatest relationships between variables, (more than 0.7 in the spearman method)": greatest_relationships_variables,
        "Descriptive statistics of the data": data_description,
        "Statistics and distribution analysis to eliminate outliers": stats_data
    }

    return exploration_results

# FastAPI route to retrieve and explore data
@app.get("/data-exploration")
def explore_data() -> dict:
    """
    FastAPI route to retrieve and explore data.

    This function retrieves data from an API, converts it to a DataFrame, and then performs data exploration and analysis
    using the `data_exploration_and_analysis` function. The results of the analysis are returned as a dictionary.

    Returns:
        dict: A dictionary containing the results of the data exploration and analysis.
    """

    # Retrieve data from the API
    data = get_data()

    # Convert the JSON object to a DataFrame
    data = pd.read_json(data)

    # Perform data exploration and analysis
    result = data_exploration_and_analysis(data)

    # Return the results of the data exploration and analysis
    return result

## -- Clustering Analysis -- ##

def preprocess_data(data):
    """
    Preprocesses the given data.

    This function removes outliers from the data and then scales the data using StandardScaler.

    Args:
        data (DataFrame): The data to preprocess.
    """

    # Remove outliers from the data
    data_transformed = remove_outliers(data)

    # Initialize a StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    data_scaled = scaler.fit_transform(data_transformed)

    # Return the scaled data and the data with outliers removed
    return data_scaled, data_transformed

def tune_hyperparameters(data_scaled: np.ndarray) -> dict:
    """
    Tunes hyperparameters for KMeans using GridSearchCV.

    Args:
        data_scaled (numpy.ndarray): The scaled data.

    Returns:
        dict: The best parameters found by GridSearchCV.
    """
    n_init_range = list(range(1, 10))
    param_grid = {'n_init': n_init_range,
                  'init': ['k-means++', 'random'],
                  'max_iter': [100, 200, 300, 400, 500, 600]}

    grid = GridSearchCV(KMeans(), param_grid, cv=10, scoring='accuracy', refit='silhouette_score')
    grid.fit(data_scaled)

    # Best parameters selected in the grid search
    best_params = grid.best_params_

    return best_params

def train_model(data_scaled: np.ndarray, best_params: dict):
    """
    Trains a KMeans model using the best parameters.

    Args:
        data_scaled (numpy.ndarray): The scaled data.
        best_params (dict): The best parameters for KMeans.

    Returns:
        tuple: The predicted labels, silhouette score, and inertia score.
    """
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, **best_params, random_state=42)
    pred_y = kmeans.fit_predict(data_scaled)

    # Calculating silhouette score for KMeans model
    score = silhouette_score(data_scaled, pred_y)
    inertia_score = kmeans.inertia_

    return pred_y, score, inertia_score

def cluster_analysis(data_transformed, pred_y):
    """
    Performs cluster analysis on the transformed data.

    This function adds the predicted labels to the data, calculates the mean and standard deviation of each cluster, 
    and counts the number of data points in each cluster.

    Args:
        data_transformed (DataFrame): The transformed data.
        pred_y (numpy.ndarray): The predicted labels.

    Returns:
        tuple: A tuple containing the means, standard deviations, and sizes of the clusters.
    """
    # Add the predicted labels to the data
    data_transformed['Cluster'] = pred_y

    # Calculate the mean and standard deviation of each cluster
    cluster_means = data_transformed.groupby('Cluster').mean()
    cluster_std = data_transformed.groupby('Cluster').std()

    # Count the number of data points in each cluster
    cluster_sizes = data_transformed['Cluster'].value_counts()

    return cluster_means, cluster_std, cluster_sizes

@app.get("/clustering-analysis")
def perform_clustering_analysis() -> dict:
    """
    FastAPI route to perform clustering analysis.

    This function retrieves data, preprocesses it, tunes hyperparameters, trains a model, performs cluster analysis, 
    and returns the results.

    Returns:
        dict: A dictionary containing the results of the clustering analysis.
    """
    # Retrieve and preprocess data
    data = pd.read_json(get_data())
    data_scaled, data_transformed = preprocess_data(data)

    # Tune hyperparameters and train model
    best_params = tune_hyperparameters(data_scaled)
    pred_y, score, inertia_score = train_model(data_scaled, best_params)

    # Perform cluster analysis
    cluster_means, cluster_std, cluster_sizes = cluster_analysis(data_transformed, pred_y)

    # Dictionary with all the results
    clustering_results = {
        "The optimal number of clusters have been set to 3": "It was selected through the elbow method and silhouette score visualizations.",
        "Best parameters": best_params,
        "Silhouette score": round(score, 2),
        "Inertia score": round(inertia_score, 2),
        "Cluster sizes": cluster_sizes.to_dict(),
        "Cluster means": cluster_means.to_dict(),
        "Cluster standard deviations": cluster_std.to_dict(),
    }

    return clustering_results

# Run the Flask application
if __name__ == '__main__':
    run(app, port=8000, host='0.0.0.0', log_level='info')