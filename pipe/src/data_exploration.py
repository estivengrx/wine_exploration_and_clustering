# data_exploration_script.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_extraction_api

from warnings import filterwarnings
from requests import get
from io import StringIO
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from ydata_profiling import ProfileReport
from matplotlib import patches as mpatches


def data_from_api():
    data = data_extraction_api.get_data()
    data = pd.read_json(data)
    return data

def data_exploration(data):
    print(f'Dataframe dimensions: {data.shape}')

    print(f'Total of null data: {data.isnull().sum().sum()}')
    print(f'Total of duplicated data: {data.duplicated().sum()}')

    inf_values = np.isinf(data)
    inf_rows, inf_cols = np.where(inf_values)

    print(f'Rows with inf values: {inf_rows.sum()}')
    print(f'Columns with inf values: {inf_cols.sum()}')

    print(f'Data information: \n{data.info()}')

    print(f'Data description: \n{data.describe()}')

    # Full distribution analysis function
    def calculate_stats(data: pd.DataFrame) -> list:
        """
        Calculate the mean, standard deviation, skewness, and kurtosis for each column in a DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame for which to calculate statistics.

        Returns:
        mean (list): The mean of each column.
        std (list): The standard deviation of each column.
        skewness (list): The skewness of each column.
        kurtosis (list): The kurtosis of each column.
        """
    
        # Initialize lists to store results
        mean, std, skewness, kurtosis = [], [], [], []
        
        for column in data.columns:
            mean.append(data[column].mean())
            std.append(data[column].std())
            skewness.append(stats.skew(data[column]))
            kurtosis.append(stats.kurtosis(data[column]))
        
        # Return the calculated statistics
        return mean, std, skewness, kurtosis

    # Interquartile Range (IQR) method to remove outliers
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75) # First and third quartiles
    IQR = Q3 - Q1
    data_transformed = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)] # Remove outliers applying IQR method
    # Full distribution analysis with and without outliers
    mean, std, skewness, kurtosis = calculate_stats(data)
    mean_no_outliers, std_no_outliers, skewness_no_outliers, kurtosis_no_outliers = calculate_stats(data_transformed)

    stats_data = pd.DataFrame({
        'Column': data.columns, 
        'Mean': mean,
        'Mean no outliers': mean_no_outliers,
        'Std': std,
        'Std no outliers': std_no_outliers,
        'Skewness': skewness,
        'Skewness no outliers': skewness_no_outliers,
        'Kurtosis': kurtosis,
        'Kurtosis no outliers': kurtosis_no_outliers
    })

    print(stats_data)

    # Number of rows deleted
    rows_deleted = len(data) - len(data_transformed)
    print(f'Number of rows deleted: {rows_deleted}')
    print(f'Percentage of rows deleted: {rows_deleted / len(data) * 100:.1f}%')

    # Pairplot to visualize relationships between variables
    sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
    plt.suptitle("Pairplot of the data", y=1.02, size=20)
    plt.show()

    # Boxplots for each variable affected by the outlier treatment
    sns.set(font_scale=1.8)
    columns_affected = ['Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Proanthocyanins', 'Color_Intensity', 'Hue']

    num_plots = len(data[columns_affected].columns)
    total_cols = 3
    total_rows = num_plots // total_cols + 1
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                            figsize=(7*total_cols, 7*total_rows), constrained_layout=True)

    flierprops_orig = dict(marker='o', markerfacecolor='red', markersize=6,
                           linestyle='none', markeredgecolor='red')
    flierprops_transformed = dict(marker='o', markerfacecolor='blue', markersize=6,
                                  linestyle='none', markeredgecolor='blue')

    for i, var in enumerate(data[columns_affected].columns):
        row = i // total_cols
        pos = i % total_cols
        plot = sns.boxplot(x=var, data=data[columns_affected], ax=axs[row][pos], flierprops=flierprops_orig, color='lightblue')
        plot = sns.boxplot(x=var, data=data_transformed[columns_affected], ax=axs[row][pos], flierprops=flierprops_transformed, color='#78ACA5')
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

    profile_report = ProfileReport(data_transformed, title='Wine clustering profiling report',
                               explorative=True, correlations={"spearman": {"calculate": True}})

    profile_report.to_file('../reports/wine_clustering_profiling_report.html')
    profile_report.to_notebook_iframe()

    matrix_correlations_from_report = profile_report._description_set.correlations['spearman']

    # Mask to search for the most important relationships
    mask = np.abs(matrix_correlations_from_report.values) > 0.8

    # Get the relationships
    relationships = [(matrix_correlations_from_report.columns[i], 
                    matrix_correlations_from_report.columns[j]) \
                    for i, j in zip(*np.where(mask)) if i != j]

    # Convert each inner list to a tuple
    relationships_tuples = [tuple(sorted(rel)) for rel in relationships]

    # Use a set to remove duplicates
    unique_relationships = list(set(relationships_tuples))
    print(f'Relationships with a coefficient grater than 0.7: {unique_relationships}')

if __name__ == "__main__":
    # Set up seaborn styling
    sns.set(font_scale=0.7)

    # Ignore warnings
    filterwarnings('ignore')
    data = data_from_api()
    data_exploration(data)
