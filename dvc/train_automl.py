import yaml
import pandas as pd
import os
from autogluon.tabular import TabularPredictor

def train_automl():
    # Create directories if they don't exist
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../metrics', exist_ok=True)

    print(f"Working directory: {os.getcwd()}")
    print(f"Model will be saved to: {os.path.abspath('../models/automl_model')}")

    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load training data
    train_data = pd.read_csv('../data/prepared/train.csv')
    
    # Get label column name
    label_column = list(params['goal_features'].values())[0]['feature']
    
    # Remove rows with NaN or infinite values in the label column
    train_data = train_data.dropna(subset=[label_column])
    train_data = train_data[~train_data[label_column].isin([float('inf'), float('-inf')])]
    
    # Get AutoML parameters
    automl_params = next(model for model in params['ml_models'] if model['type'] == 'AutoML')

    # Train model using AutoGluon
    predictor = TabularPredictor(label=label_column, path='../models/automl_model')
    predictor.fit(train_data, **automl_params['parameters'])

    # Save leaderboard
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv('../metrics/automl_leaderboard.csv', index=False)
    print(f"Leaderboard saved successfully: {os.path.exists('../metrics/automl_leaderboard.csv')}")

if __name__ == "__main__":
    train_automl()
