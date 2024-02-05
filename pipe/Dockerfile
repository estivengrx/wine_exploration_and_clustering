# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /clustering-wine-docker

# Copy the current directory contents into the container at /app
COPY ./src ./src
COPY requirements_scripts.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements_scripts.txt

# Run app.py when the container launches
CMD python python ./src/data_exploration.py && python ./src/clustering_analysis.py