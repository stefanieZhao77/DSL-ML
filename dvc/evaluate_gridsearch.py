import yaml
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import json

def evaluate_gridsearch():
    # Load parameters
    with open('dvc/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load test data
    test_data = pd.read_csv('data/test/test.csv')

    # Load models
    best_models = joblib.load('models/gridsearch_model')

    features = [feature for dataset in params['features'].values() for feature in dataset]
    target = list(params['goal_features'].values())[0][0]

    X_test = test_data[features]
    y_test = test_data[target]

    metrics = {}
    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)

        if params['model']['task'] == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics[model_name] = {'mse': mse, 'r2': r2}
        else:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            metrics[model_name] = {'accuracy': accuracy, 'f1': f1}

    # Save metrics
    with open('metrics/gridsearch_evaluation.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_gridsearch()
