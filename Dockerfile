# This Dockerfile is meant to be run from the root of the repo

# Base Python image (Python 3.12)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current folder (on the host machine) that contains the whole app project into the container /app folder
COPY . /app

# To be able to install the package lgbm
RUN apt-get update && apt-get install -y \
    gcc \
    libgcc1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN dpkg -s libgomp1

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which FastAPI will run
EXPOSE 8000

# Set environment variables
#ENV DSBA_MODELS_ROOT_PATH=/app/src/adclick/models_registry
ENV PYTHONPATH=/app/src
ENV DSBA_MODELS_ROOT_PATH=/app/src/adclick/model_registry

# Define the default command run when starting the container: Run the FastAPI app using Uvicorn
#CMD ["python", "-m","uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-m", "uvicorn", "adclick.main:app", "--host", "0.0.0.0", "--port", "8000"]

