import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlite3

def handle_features(feature_name, datasets, features, dataset_names, start, end, context, dataset1, feature1, dataset2, feature2):
    selected_features = []
    id_list = []
    id_map = {}
    if dataset1 and feature1 and dataset2 and feature2:
        for n in range(len(dataset1)):
            if dataset1[n] not in id_map and dataset2[n] not in id_map:             
                id_map[dataset1[n]] = feature1[n]
                id_map[dataset2[n]] = feature2[n]

        first_key, first_value = list(id_map.items())[0]
        dataset = datasets[first_key]
        id_list = list(dataset[first_value])
        id_list = list(set(id_list))
    if features:
        for feature in features:
            feature_values = []
            if len(id_list)!=0 and feature.dataset in id_map.items():
                datasets[feature.dataset].set_index(id_map[feature.dataset], inplace=True)
                datasets[feature.dataset] = datasets[feature.dataset].reindex(id_list)
                datasets[feature.dataset].reset_index(inplace=True)
            if feature.operator and feature.value:
                for feature_value in datasets[feature.dataset][feature.feature]:
                    if eval(f"{feature_value} {feature.operator} {feature.value}"):
                        feature_values.append(feature_value)                
                    else:
                        feature_values.append(None)
                selected_features.append(pd.Series(feature_values, name=feature.feature))
            elif feature.feature != "All":
                selected_features.append(datasets[feature.dataset][feature.feature])
            elif feature.feature == "All":
                all_features = datasets[feature.dataset].columns
                feature_counts = all_features.value_counts()
                duplicated_features = feature_counts[feature_counts > 1].index.tolist()
                for duplicated_feature in duplicated_features:
                    for i in range(len(all_features)):
                        if all_features[i] == duplicated_feature:
                            all_features[i] = f"{feature.dataset}_{duplicated_feature}"
                selected_features.append(datasets[feature.dataset][all_features])

    if start and end:
        start = start[0]
        end = end[0]
        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]
            if start > dataset.columns.size or end > dataset.columns.size:
                raise ValueError("Start and end values must be less than the number of rows in the dataset")
            elif start < 0 or end < 0:
                raise ValueError("Start and end values must be greater than 0")
            elif start > end:
                raise ValueError("Start value must be less than or equal to end value")
            sliced_rows = dataset.iloc[start:end+1]
            selected_features.append(sliced_rows)
 

    # Concatenate selected features
    concatenated_features = pd.concat(selected_features, axis=1)
    concatenated_features = concatenated_features.dropna()
    context[feature_name] = concatenated_features
    
    return context

def reduce_features(data_set, value):
    reduced_data_set = data_set.head(value)

    return reduced_data_set
      
def load_dataset(name, path, datasets):
    if path.endswith('.csv'):
        datasets[name] = pd.read_csv(path)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        datasets[name] = pd.read_excel(path)
    elif path.endswith('.db'):
        conn = sqlite3.connect(path)
        datasets[name] = pd.read_sql_query("SELECT * FROM " + name, conn)
        conn.close()

def train_data(feature_list, target_feature, test_size=0.2, random_state=42):
    scaler = StandardScaler()
    features = feature_list.drop(target_feature, axis=1).dropna()
    X = scaler.fit_transform(features)
    y = feature_list[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_data = {}
    train_data["X_train"] = X_train
    train_data["X_test"] = X_test
    train_data["y_train"] = y_train
    train_data["y_test"] = y_test
    return train_data

    
