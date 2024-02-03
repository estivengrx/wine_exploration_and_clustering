# Import necessary libraries

from uvicorn import run
from fastapi import FastAPI
from requests import get
from io import StringIO
from pandas import read_csv

# Initialize the Flask application
app = FastAPI()

@app.get('/')
def get_data():
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv' # Direct link to the dataset
    response = get(url)

    if response.status_code == 200:
        data = read_csv(StringIO(response.text)) # Dataframe from the csv url
        print('Data retrieved successfully')
    else:
        print(f'Failed to download dataset. Status code: {response.status_code}')
    return data.to_json() # Return the data as a JSON object
    

# Run the Flask application
if __name__ == '__main__':
    run(app, port=8000, host='127.0.0.1')