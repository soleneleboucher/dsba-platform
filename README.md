# Machine Learning CLI Tool

## Overview
This CLI tool allows users to train, evaluate, and predict using machine learning models on structured data. It supports logistic regression, random forests, and LightGBM models.

## Installation

### Without Docker
Ensure you have the required dependencies installed:
```bash
pip install -r requirement.txt
```

### With Docker

Alternatively, you can run the tool using Docker. This will automatically set up the environment and dependencies for you.

### Deploy Docker directly on your machine 

##### 1. Build the Docker Image: First, build the Docker image:

```bash
docker build -t ml-cli-tool .
```

##### 2. Run the Docker Container: Start the container and expose port 8000:

```bash
docker run -p 8000:8000 ml-cli-tool
```
This will start the FastAPI server that you can interact with via your browser or API requests.

### Deploy the Docker image on Docker Hub 

In addition to running Docker locally, you can push your image to Docker Hub, which allows to share and deploy the application. 

For that, the docker_push.sh file contains all the necessary commands. 

You need to execute: 

- to grant the rights to execute: 

```bash
chmod +x docker_push.sh
```

- to execute the script:

```bash
./docker_push.sh
```

## Usage
Run the `main.py` script using the command line:

### 1. Train a Model
```bash
python -m src.adclick.main --data path/to/data.csv --task train --model logistic_regression
```

### 2. Evaluate a Model
```bash
python -m src.adclick.main --data path/to/data.csv --task evaluate --model logistic_regression
```

### 3. Make Predictions
```bash
python -m src.adclick.main --task predict --model logistic_regression --predict_data path/to/new_data.csv
```


## Options

### --task
Specifies the operation to perform. Available options:
- `train` – Train the specified model.
- `evaluate` – Evaluate the performance of the trained model.
- `predict` – Make predictions using the trained model.

### --model
Specifies which machine learning model to use. Available options:
- `logistic_regression` – Logistic Regression model.
- `random_forest` – Random Forest model.
- `lightgbm` – LightGBM model.

## File Structure
```bash
project-root/
│
├── src/
│   └── adclick/
│       ├── __init__.py
│       ├── main.py
│       ├── dataloader.py
│       ├── model_training.py
│       ├── model_evaluation.py
│       ├── model_prediction.py
│       ├── model_registry.py
│       ├── models_registry/ # where trained models are saved 
│
├── docker_push.sh
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Notes
- Ensure models are trained before running predictions.
- Models are saved in the `models_registry/` directory.

