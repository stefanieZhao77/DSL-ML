import yaml
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import json

def evaluate_automl():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load test data
    test_data = pd.read_csv('../data/test/test.csv')

    # Load model
    predictor = TabularPredictor.load('../models/automl_model')

    # Extract feature names correctly
    features = []
    for dataset_features in params['features'].values():
        features.extend([feature['name'] for feature in dataset_features])
    
    # Add engineered features if they exist
    if 'feature_engineering' in params:
        for operation in params['feature_engineering']:
            if operation['type'] == 'create':
                features.append(operation['new_feature'])

    target = params['goal_features']['features']['feature']

    X_test = test_data[features]
    y_test = test_data[target]

    # Make predictions
    y_pred = predictor.predict(X_test)

    # Remove rows where either y_test or y_pred contains NaN
    mask = ~(y_test.isna() | pd.Series(y_pred).isna())
    y_test = y_test[mask]
    y_pred = y_pred[mask]

    # Calculate metrics
    metrics = {}
    for metric_name in params['metrics']:
        if metric_name == 'rmse':
            metrics['rmse'] = float(mean_squared_error(y_test, y_pred, squared=False))
        elif metric_name == 'mae':
            metrics['mae'] = float(mean_squared_error(y_test, y_pred))
        elif metric_name == 'r2_score':
            metrics['r2_score'] = float(r2_score(y_test, y_pred))
    
    # Save metrics
    with open('../metrics/automl_evaluation.json', 'w') as f:
        json.dump(metrics, f)
    # Get feature importance
    feature_importance = predictor.feature_importance(test_data)
    
    # Get list of features from params
    features = []
    # Add original features
    for feature_info in params['features']['mrt_data']:
        features.append(feature_info['name'])
    # Add created features
    for operation in params['feature_engineering']:
        if operation['type'] == 'create':
            features.append(operation['new_feature'])
    
    # Create feature importance DataFrame
    try:
        if isinstance(feature_importance, pd.DataFrame):
            # If it's already a DataFrame, ensure features are properly labeled
            feature_importance_df = feature_importance.copy()
            feature_importance_df['feature'] = features
        else:
            # Create DataFrame with feature names
            importance_values = feature_importance.values.flatten() if hasattr(feature_importance, 'values') else feature_importance
            
            print(f"Number of features: {len(features)}")
            print(f"Number of importance values: {len(importance_values)}")
            
            assert len(features) == len(importance_values), "Feature names and importance values must have same length"
            
            feature_importance_df = pd.DataFrame({
                'feature': features,
                'importance': importance_values,
                'stddev': feature_importance.get('stddev', [0] * len(features)),
                'p_value': feature_importance.get('p_value', [0] * len(features)),
                'n': feature_importance.get('n', [5] * len(features)),
                'p99_high': feature_importance.get('p99_high', [0] * len(features)),
                'p99_low': feature_importance.get('p99_low', [0] * len(features))
            })
        
        # Save feature importance
        feature_importance_df.to_csv('../metrics/feature_importance.csv', index=False)
        
    except Exception as e:
        print(f"Warning: Could not save feature importance due to error: {str(e)}")
        print("Feature importance object:", feature_importance)
        # Continue execution even if feature importance fails
        pass

if __name__ == "__main__":
    evaluate_automl()
