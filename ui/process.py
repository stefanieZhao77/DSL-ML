from textx import metamodel_from_file, textx_isinstance
import yaml
import os
import shutil
import subprocess

def execute_dsl(dsl_content):
    try:
        if not _write_all_to_params(dsl_content):
            return False, "Error writing all to params"
        if not _update_dvc_pipeline():
            return False, "Error updating DVC pipeline"
        if not _run_dvc_pipeline():
            return False, "Error running DVC pipeline"
        return True, "DSL executed successfully, and ML Training is working."
    except Exception as e:
        return False, f"Error executing DSL: {str(e)}"            


def _write_all_to_params(dsl_content):
    try:
        mm = metamodel_from_file('rule_model.tx')
        my_model = mm.model_from_str(dsl_content)
        
        # Initialize containers for extracted data
        models = []
        ml_models = []
        features_by_dataset = {}
        metrics = []
        goal_features = {}
        file_mapping = {}
        feature_engineering = []
        start = {}

        # Extract all data in a single loop
        for element in my_model.elements:
            if textx_isinstance(element, mm['Load']):
                _copy_files(element.path, '../data/raw')
                file_mapping[element.name] = os.path.join('../data/raw', os.path.basename(element.path))
            elif textx_isinstance(element, mm['Model']):
                model = {
                    'name': element.name,
                    'task': element.task
                }
                models.append(model)
                for model_element in element.elements:
                    if textx_isinstance(model_element, mm['MLModel']):
                        ml_models.append({
                            'name': model_element.name,
                            'type': model_element.modelType,
                            'parameters': {param.name: param.value for param in model_element.parameters}
                        })
                    elif textx_isinstance(model_element, mm['FeatureSelection']):
                        # Get the FeatureList content directly
                        feature_list = getattr(model_element, 'features', None)
                        if not feature_list:
                            continue
                            
                        # Access the actual features through the features attribute of FeatureList
                        features_to_process = getattr(feature_list, 'features', [])
                        for feature in features_to_process:
                            if hasattr(feature, 'dataset') and hasattr(feature, 'start') and hasattr(feature, 'end'):
                                # Handle range-based feature selection
                                dataset = feature.dataset
                                if dataset not in features_by_dataset:
                                    features_by_dataset[dataset] = []
                                features_by_dataset[dataset].extend(list(range(feature.start, feature.end + 1)))
                            elif hasattr(feature, 'all') and feature.all:
                                # Handle 'all' features case
                                for dataset_name, file_path in file_mapping.items():
                                    if dataset_name not in features_by_dataset:
                                        features_by_dataset[dataset_name] = 'all'
                            else:
                                dataset = feature.dataset
                                feature_name = feature.feature
                                if dataset not in features_by_dataset:
                                    features_by_dataset[dataset] = []
                                # Only append features that match the filter conditions
                                if hasattr(feature, 'operator') and hasattr(feature, 'value') and feature.operator and feature.value:
                                    # Apply filtering based on operator and value
                                    if feature.operator == '>':
                                        if float(feature.value) > 0:
                                            features_by_dataset[dataset].append({'name': feature_name})
                                    elif feature.operator == '<':
                                        if float(feature.value) < 0:
                                            features_by_dataset[dataset].append({'name': feature_name})
                                    elif feature.operator == '>=':
                                        if float(feature.value) >= 0:
                                            features_by_dataset[dataset].append({'name': feature_name})
                                    elif feature.operator == '<=':
                                        if float(feature.value) <= 0:
                                            features_by_dataset[dataset].append({'name': feature_name})
                                    elif feature.operator == '==':
                                        if float(feature.value) == 0:
                                            features_by_dataset[dataset].append({'name': feature_name})
                                    elif feature.operator == '!=':
                                        if float(feature.value) != 0:
                                            features_by_dataset[dataset].append({'name': feature_name})
                                else:
                                    # If no operator/value, include feature without filtering
                                    features_by_dataset[dataset].append({'name': feature_name})
                        
                            # Handle goal feature if present
                            if hasattr(model_element, 'goal') and model_element.goal:
                                goal = model_element.goal
                                goal_features[model_element.name] = {
                                    'dataset': goal.dataset,
                                    'feature': goal.feature
                                }
                    elif textx_isinstance(model_element, mm['Metric']):
                        metrics.extend([value for value in model_element.name.values])
                    elif textx_isinstance(model_element, mm['FeatureEngineering']):
                        for operation in model_element.operations:
                            if textx_isinstance(operation, mm['CreateFeature']):
                                feature_engineering.append({
                                    'type': 'create',
                                    'new_feature': operation.newFeature,
                                    'expression': operation.expression
                                })
                            elif textx_isinstance(operation, mm['TransformFeature']):
                                feature_engineering.append({
                                    'type': 'transform',
                                    'feature': f"{operation.feature.dataset}.{operation.feature.feature}",
                                    'method': operation.method
                                })
                            elif textx_isinstance(operation, mm['EncodeFeature']):
                                feature_engineering.append({
                                    'type': 'encode',
                                    'feature': f"{operation.feature.dataset}.{operation.feature.feature}",
                                    'method': operation.method
                                })
                    elif textx_isinstance(model_element, mm['Start']):
                        start_model = model_element.mlModels[0]
                        start_feature = model_element.feature
                        start[start_model] = start_feature

        # Read existing params.yaml
        params_file = 'dvc/params.yaml'
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                existing_params = yaml.safe_load(f)
        else:
            existing_params = {}

        # Update existing params with new data
        existing_params['models'] = models
        existing_params['ml_models'] = ml_models
        existing_params['features'] = features_by_dataset
        existing_params['metrics'] = metrics
        existing_params['goal_features'] = goal_features
        existing_params['file_mapping'] = file_mapping
        existing_params['feature_engineering'] = feature_engineering
        existing_params['start'] = start
        # Write updated params back to params.yaml
        with open(params_file, 'w') as f:
            yaml.dump(existing_params, f, default_flow_style=False)

        print("All parameters updated in params.yaml successfully")
        return True, "All parameters updated in params.yaml successfully"
    except Exception as e:
        return False, f"Error updating parameters in params.yaml: {str(e)}"



def _copy_files(src, dst):
    try:
        dst_file = os.path.join(dst, os.path.basename(src))
        if not os.path.exists(dst_file):
            print(f"Destination file does not exist: {dst_file}")
            shutil.copy(src, dst)
            print(f"File copied from {src} to {dst}")
    except Exception as e:
        print(f"Error copying file: {e}")

def _update_dvc_pipeline():
    # Read params.yaml
    with open('dvc/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Define plots configuration
    plots = [
        {
            '../metrics/automl_leaderboard.csv': {
                'x': 'model',
                'y': 'score_val',
                'title': 'AutoML Model Performance',
                'template': 'bar',
                'x_label': 'Model Type',
                'y_label': 'Validation Score'
            }
        },
        {
            '../metrics/feature_importance.csv': {
                'x': 'importance',
                'y': 'feature',
                'title': 'Feature Importance Analysis',
                'template': 'scatter',
                'x_label': 'Importance Score',
                'y_label': 'Features',
                'error_y': 'stddev',  # Add error bars using stddev
                'error_x': None
            }
        },
        {
            '../metrics/automl_evaluation.json': {
                'template': 'simple',
                'title': 'AutoML Model Evaluation Metrics',
                'metrics': ['rmse', 'mae', 'r2_score']
            }
        },
        {
            '../metrics/evaluation_metrics.json': {
                'template': 'simple',
                'title': 'Overall Model Evaluation',
                'metrics': ['root_mean_squared_error', 'r2', 'pearsonr']
            }
        }
    ]

    # Create DVC pipeline with updated plots
    dvc_yaml = {
        'stages': {},
        'plots': []  # Changed to list format
    }

    # Add stages (unchanged)
    dvc_yaml['stages']['prepare'] = {
        'cmd': 'python prepare.py',
        'deps': ['../data/raw', 'prepare.py'],
        'outs': ['../data/prepared', '../data/test']
    }

    # Add training stages next
    if any(model['type'] in ['RandomForest', 'DecisionTree', 'SVM'] for model in params['ml_models']):
        dvc_yaml['stages']['train_gridsearch'] = {
            'cmd': 'python train_gridsearch.py',
            'deps': ['../data/prepared/train.csv', 'train_gridsearch.py'],
            'outs': ['../models/gridsearch_model'],
            'metrics': ['../metrics/gridsearch_results.csv']
        }

    if 'AutoML' in [model['type'] for model in params['ml_models']]:
        dvc_yaml['stages']['train_automl'] = {
            'cmd': 'python train_automl.py',
            'deps': ['../data/prepared/train.csv', 'train_automl.py'],
            'outs': ['../models/automl_model'],
            'metrics': ['../metrics/automl_leaderboard.csv']
        }

    # Add evaluation stages last
    if any(model['type'] in ['RandomForest', 'DecisionTree', 'SVM'] for model in params['ml_models']):
        dvc_yaml['stages']['evaluate_gridsearch'] = {
            'cmd': 'python evaluate_gridsearch.py',
            'deps': ['../data/test/test.csv', 'evaluate_gridsearch.py', 'models/gridsearch_model'],
            'metrics': ['../metrics/gridsearch_evaluation.json']
        }

    if 'AutoML' in [model['type'] for model in params['ml_models']]:
        dvc_yaml['stages']['evaluate_automl'] = {
            'cmd': 'python evaluate_automl.py',
            'deps': ['../data/test/test.csv', 'evaluate_automl.py', 'models/automl_model'],
            'metrics': ['../metrics/automl_evaluation.json']
        }

    # Update plots to be a list format
    dvc_yaml['plots'] = [
        {
            '../metrics/automl_leaderboard.csv': {
                'x': 'model',
                'y': 'score_val',
                'title': 'AutoML Model Performance',
                'template': 'bar',
                'x_label': 'Model Type',
                'y_label': 'Validation Score'
            }
        },
        {
            '../metrics/feature_importance.csv': {
                'x': 'importance',
                'y': 'feature',
                'title': 'Feature Importance Analysis',
                'template': 'scatter',
                'x_label': 'Importance Score',
                'y_label': 'Features',
            }
        },
        {
            '../metrics/automl_evaluation.json': {
                'template': 'simple',
                'title': 'AutoML Model Evaluation Metrics',
                'y': ['rmse', 'mae', 'r2_score']
            }
        },
        {
            '../metrics/evaluation_metrics.json': {
                'template': 'simple',
                'title': 'Overall Model Evaluation',
                'y': ['root_mean_squared_error', 'r2', 'pearsonr']
            }
        }
    ]

    # Add clustering stage if clustering models are specified
    if any(model['type'] in ['KMeans', 'DBSCAN', 'AgglomerativeClustering'] for model in params['ml_models']):
        dvc_yaml['stages']['train_cluster'] = {
            'cmd': 'python train_cluster.py',
            'deps': ['../data/prepared/train.csv', 'train_cluster.py'],
            'outs': ['../models/cluster_model_*', '../models/cluster_scaler_*'],
            'metrics': [
                '../metrics/clustering_metrics.json',
                '../metrics/cluster_assignments_*.csv',
                '../metrics/feature_importance_*.csv'
            ]
        }

    # Write updated DVC pipeline
    with open('dvc/dvc.yaml', 'w') as f:
        yaml.dump(dvc_yaml, f, default_flow_style=False, sort_keys=False)

    return True, "DVC pipeline updated successfully."

def _run_dvc_pipeline():
    try:
        subprocess.run(['dvc', 'repro', 'dvc/dvc.yaml'], check=True)
        return True, "DVC pipeline executed successfully."
    except subprocess.CalledProcessError as e:
        return False, f"Error executing DVC pipeline: {e}"
        
