import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import numpy as np
import os

def apply_feature_engineering(data, params):
    for operation in params['feature_engineering']:
        if operation['type'] == 'create':
            expression = operation['expression']
            # Replace all dataset prefixes in the expression
            for col in data.columns:
                for dataset_prefix in params['file_mapping'].keys():
                    expression = expression.replace(f"{dataset_prefix}.", "")
            try:
                data[operation['new_feature']] = data.eval(expression)
            except Exception as e:
                print(f"Error creating feature {operation['new_feature']}: {str(e)}")
                continue
                
        elif operation['type'] == 'transform':
            # Extract feature name without dataset prefix
            dataset, feature = operation['feature'].split('.')
            
            # Check if the feature exists in the dataframe
            if feature not in data.columns:
                print(f"Warning: Feature {feature} not found in columns: {data.columns}")
                continue

            try:
                if operation['method'] == 'log':
                    data[feature] = np.log1p(data[feature])
                elif operation['method'] == 'sqrt':
                    data[feature] = np.sqrt(data[feature])
                elif operation['method'] == 'square':
                    data[feature] = np.square(data[feature])
                elif operation['method'] == 'standardize':
                    scaler = StandardScaler()
                    data[feature] = scaler.fit_transform(data[[feature]])
                elif operation['method'] == 'normalize':
                    scaler = MinMaxScaler()
                    data[feature] = scaler.fit_transform(data[[feature]])
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
                continue

        elif operation['type'] == 'encode':
            dataset, feature = operation['feature'].split('.')
            
            # Check if the feature exists in the dataframe
            if feature not in data.columns:
                print(f"Warning: Feature {feature} not found in columns: {data.columns}")
                continue

            try:
                if operation['method'] == 'onehot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(data[[feature]])
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=[f"{feature}_{cat}" for cat in encoder.categories_[0]]
                    )
                    data = pd.concat([data, encoded_df], axis=1)
                elif operation['method'] == 'label':
                    encoder = LabelEncoder()
                    data[f"{feature}_encoded"] = encoder.fit_transform(data[feature])
                elif operation['method'] == 'frequency':
                    freq_encoding = data[feature].value_counts(normalize=True)
                    data[f"{feature}_freq_encoded"] = data[feature].map(freq_encoding)
            except Exception as e:
                print(f"Error encoding feature {feature}: {str(e)}")
                continue

    return data

def prepare_data():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load datasets
    datasets = {}
    for dataset_name, file_path in params['file_mapping'].items():
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            datasets[dataset_name] = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            datasets[dataset_name] = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    # Get id_mapping from params
    id_mapping = params['data']['id_mapping']

    # Merge datasets based on id_mapping
    merged_data = None
    for dataset_name, id_column in id_mapping.items():
        if merged_data is None:
            merged_data = datasets[dataset_name].set_index(id_column)
        else:
            merged_data = merged_data.join(datasets[dataset_name].set_index(id_column), how='outer')

    # Reset index to make the ID column a regular column
    merged_data.reset_index(inplace=True)

    # Debug print
    print("Available columns:", merged_data.columns.tolist())

    # Apply feature engineering if specified
    if 'feature_engineering' in params and params['feature_engineering']:
        merged_data = apply_feature_engineering(merged_data, params)

    # Extract features (removing dataset prefixes if present)
    features = []
    for dataset_features in params['features'].values():
        for feature in dataset_features:
            if isinstance(feature, dict) and 'name' in feature:
                feature_name = feature['name']
            else:
                feature_name = feature
            # Only add feature if it's not already in the list
            if feature_name not in features:
                features.append(feature_name)
    
    # Add engineered features to the features list
    if 'feature_engineering' in params:
        for operation in params['feature_engineering']:
            if operation['type'] == 'create':
                new_feature = operation['new_feature']
                if new_feature not in features:
                    features.append(new_feature)
            elif operation['type'] == 'transform':
                _, feature = operation['feature'].split('.')
                # Don't add transformed feature if it's already in the list
                if feature not in features:
                    features.append(feature)

    # Get target variable (removing dataset prefix if present)
    target_info = list(params['goal_features'].values())[0]
    target = target_info['feature']

    # Remove any duplicates from features list if they still exist
    features = list(dict.fromkeys(features))

    X = merged_data[features]
    y = merged_data[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['data']['test_size'], random_state=params['data']['random_state'])

    # Save prepared datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Create directories if they don't exist
    os.makedirs('../data/prepared', exist_ok=True)
    os.makedirs('../data/test', exist_ok=True)

    train_data.to_csv('../data/prepared/train.csv', index=False)
    test_data.to_csv('../data/test/test.csv', index=False)

if __name__ == "__main__":
    prepare_data()
