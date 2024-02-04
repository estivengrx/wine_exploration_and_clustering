# data_exploration_script_with_api.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_extraction_api
from warnings import filterwarnings
from scipy import stats
from ydata_profiling import ProfileReport
from matplotlib import patches as mpatches
from fastapi import FastAPI
from io import StringIO
from contextlib import redirect_stdout

app = FastAPI()

# FastAPI route to retrieve and explore data
@app.get("/data-exploration")
def explore_data():
    # Retrieve data from the API
    data = data_extraction_api.get_data()
    data = pd.read_json(data)  # Convert the JSON object to a DataFrame

    # Perform data exploration and analysis
    result = data_exploration_and_analysis(data)

    return result

def calculate_stats(data: pd.DataFrame) -> dict:
    mean, std, skewness, kurtosis = [], [], [], []

    for column in data.columns:
        mean.append(data[column].mean())
        std.append(data[column].std())
        skewness.append(stats.skew(data[column]))
        kurtosis.append(stats.kurtosis(data[column]))

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

# Entry point of the script
if __name__ == "__main__":
    import uvicorn

    # Run FastAPI application using uvicorn
    uvicorn.run(app, port=8000, host='127.0.0.1', log_level="info")