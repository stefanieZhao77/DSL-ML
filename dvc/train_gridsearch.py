import yaml
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
import joblib

def train_gridsearch():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load training data
    train_data = pd.read_csv('../data/prepared/train.csv')
    
    features = [feature for dataset in params['features'].values() for feature in dataset]
    target = list(params['goal_features'].values())[0][0]

    X = train_data[features]
    y = train_data[target]

    # Get GridSearch parameters
    gridsearch_models = [model for model in params['ml_models'] if model['type'] in ['RandomForest', 'DecisionTree', 'SVM']]

    best_models = {}
    for model in gridsearch_models:
        if model['type'] == 'RandomForest':
            estimator = RandomForestRegressor() if params['model']['task'] == 'regression' else RandomForestClassifier()
        elif model['type'] == 'DecisionTree':
            estimator = DecisionTreeRegressor() if params['model']['task'] == 'regression' else DecisionTreeClassifier()
        elif model['type'] == 'SVM':
            estimator = SVR() if params['model']['task'] == 'regression' else SVC()

        grid_search = GridSearchCV(estimator, model['parameters'], cv=5, n_jobs=-1)
        grid_search.fit(X, y)

        best_models[model['type']] = grid_search.best_estimator_

    # Save best models
    joblib.dump(best_models, '../models/gridsearch_model')

    # Save GridSearch results
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv('../metrics/gridsearch_results.csv', index=False)

if __name__ == "__main__":
    train_gridsearch()
