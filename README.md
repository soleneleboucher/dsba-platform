# Ad Click Prediction Tool

## Table of Contents

- [Overview](#Overview)
- [Data Requirements](#Data-Requirements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Options](#Options)
- [File Structure](#File-Structure)
- [Notes](#Notes)

Download the project specifications in .docx format [here](./MLMOps%Project%Specifications.docx)

## Overview
This ad-click prediction tool, written in Python, allows companies to predict if customers will engage with a specific type of advertisement shown to them based on the customer’s demographic and online behavior. It can be used by e-commerce platforms to fine-tune their marketing campaigns and advertising strategies. The tool allows the user to select from three types of prediction models, train the model on historical data and use the trained model(s) to predict customer behavior for planned advertisements.

The models trained are saved within [models_registry](./src/adclick/models_registry/)

## Data-Requirements

The data for training and prediction is to be supplied in a standard .csv format with the following structure:

| Column Name         | Type     | Description                                                       |
|---------------------|----------|-------------------------------------------------------------------|
| `customer_region`   | string   | Geographic region of the customer                                 |
| `time_of_day`       | float    | Time of day (e.g., 13.5 for 1:30 PM)                              |
| `day_of_week`       | string   | Day name (e.g., "Monday")                                         |
| `device_type`       | string   | Device used by the customer (e.g., "mobile", "desktop")           |
| `avg_time_spent`    | float    | Average time spent on the previous platform (in minutes)          |
| `previous_purchases`| integer  | Number of past purchases by the customer                          |
| `ad_channel`        | string   | Platform where the ad was shown (e.g., "Instagram", "in-app")     |

However, the models can work on any datasets as long as the table schema remains the same. In case new columns are being utilized, you may tailor the feature engineering and data preprocessing sections within [preprocessing.py](./src/adclick/preprocessing.py.py)

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

