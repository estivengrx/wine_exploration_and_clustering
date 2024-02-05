FROM python:3.11

# Set the working directory in the Docker image to /clustering-wine-docker
WORKDIR /clustering-wine-docker

# Copy the current directory contents into the container at /clustering-wine-docker
COPY ./src ./src
COPY requirements_scripts.txt .

# Install any needed packages specified in requirements_scripts.txt
RUN pip install -r requirements_scripts.txt

# Expose the port the app runs on
EXPOSE 8000

# This is the command that starts the application
CMD ["python", "src/data_analysis_and_model.py"]