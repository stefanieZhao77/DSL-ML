import mlflow
import mlflow.sklearn
import pandas as pd

# Import models
from ml_models import decision_tree
from ml_models import logistic_regression
from ml_models import random_forest
from ml_models import svm
from ml_models import grid_search
from utils import data_manager
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MLModel(object):
    constructor_map = {
        "DecisionTree": decision_tree.DecisionTree,
        "LogisticRegression": logistic_regression.LogisticRegressionModel,
        "RandomForest": random_forest.RandomForest,
        "SVM": svm.SVM,
    }

    def __init__(self, parent, name, type, parameters, task):
        self.name = name
        self.type = type
        self.parameters = parameters
        self.params = {}
        self.task = task
        self.flag = False
        self.update_params(parameters)
        self.train_data = {}
        self.features = None

    def update_params(self, parameters):
        for param in parameters:
            self.params[param.name] = param.value.values
            if len(param.value.values) > 1:
                self.flag = True
            elif len(param.value.values) == 1:
                self.params[param.name] = param.value.values[0]
        if self.flag:
            self.model = grid_search.GridSearch(self.name, self.type, self.params, self.task, 8)
        else:
            self.model = self.constructor_map[self.type](self.params)

    def train(self, feature_list, experiment_id):
        with mlflow.start_run(experiment_id=experiment_id, run_name=self.name):
            self.features = feature_list.columns
            self.train_data = data_manager.train_data(
                feature_list, feature_list.columns[-1]
            )
            self.model.train(self.train_data["X_train"], self.train_data["y_train"])
            # Log model
            mlflow.sklearn.log_model(self.model, self.name + "_model")
            self.evaluate()

    def evaluate(self):
        X_test = self.train_data["X_test"]
        y_test = self.train_data["y_test"]
        predictions = self.model.predict(X_test)
        if hasattr(self.model, "best_model") and self.model.best_model is not None:
            result = permutation_importance(
                self.model.best_model, X_test, y_test, n_repeats=10, random_state=42
            )
        else:
            result = permutation_importance(
                self.model.model, X_test, y_test, n_repeats=10, random_state=42
            )
        self.visualize(result.importances_mean)
        metric = self.model.evaluate(predictions, y_test)
        # Log metrics
        for key, value in metric.items():
            mlflow.log_metric(key, value)
        mlflow.end_run()

    def visualize(self, importances):
        self.importance_plot(importances)
        if self.flag:
            self.visualize_grid_search()

    def importance_plot(self, importances):
        sorted_idx = importances.argsort()
        fig, ax = plt.subplots()
        ax.bar(range(len(sorted_idx)), importances[sorted_idx])
        ax.set_xticks(range(len(sorted_idx)))
        ax.set_xticklabels(self.features[sorted_idx], rotation=45, ha="right")
        ax.set_xlabel("Features")
        ax.set_ylabel("Permutation Importance")
        ax.set_title("Permutation Importance of Features")
        fig.tight_layout()
        plt.savefig("./figures/final_" + self.name + "_importance_wst.png", dpi=300)

    def visualize_grid_search(self):
        # Extract the hyperparameter values and scores
        cv_results = self.model.model.cv_results_
  
        param_grid = self.model.model.param_grid
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        mean_test_scores = cv_results['mean_test_score']
        # Create a 2D grid of parameter values
        param_grid_2d = np.meshgrid(*param_values)
        param_grid_2d = [arr.flatten() for arr in param_grid_2d]
        plt.figure(figsize=(10, 8))
        plt.title('Grid Search Results')
        plt.xlabel(param_names[0])
        plt.ylabel(param_names[1])
        plt.grid()
        # Plot the mean test scores
        scatter = plt.scatter(param_grid_2d[0], param_grid_2d[1], c=mean_test_scores, cmap='viridis', s=100)
        plt.colorbar(scatter, label='Mean Test Score')
        plt.savefig("./figures/" + self.name + "_grid_search_pst.png", dpi=300)

