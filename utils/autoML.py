from tpot import TPOTRegressor, TPOTClassifier
import mlflow
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from utils import data_manager
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

class AutoML:
    def __init__(self, task):
        self.task = task
        self.model = None
        self.train_data = {}
        self.features = None

    def train(self, feature_list, experiment_id):
        with mlflow.start_run(experiment_id=experiment_id, run_name="AutoML"):
            self.train_data = data_manager.train_data(feature_list, feature_list.columns[-1])
            self.features = feature_list.columns
            if self.task == 'regression':
                self.model = TPOTRegressor(generations=30, population_size=300, verbosity=2, config_dict='TPOT light', scoring='neg_mean_squared_error')
            elif self.task == 'classification':
                self.model = TPOTClassifier(generations=10, population_size=300, verbosity=2, config_dict='TPOT light')
            else:
                raise ValueError("Invalid task type. Supported tasks are 'regression' and 'classification'.")
            mlflow.log_params(self.model.get_params())
            self.model.fit(self.train_data["X_train"], self.train_data["y_train"])
            print(self.model.fitted_pipeline_)
            self.model.export('tpot_exported_pipeline.py')
            mlflow.sklearn.log_model(self.model.fitted_pipeline_, "AutoML_model")
            self.evaluate()
            

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")
        X_test = self.train_data["X_test"]
        y_test = self.train_data["y_test"]
        predictions = self.model.predict(X_test)
        score = self.model.score(X_test, y_test) 
        mlflow.log_metrics({'score': score, 'mse': mean_squared_error(y_test, predictions), 'r2_score': r2_score(y_test, predictions), 'mae': mean_absolute_error(y_test, predictions), 'rmse': np.sqrt(mean_squared_error(y_test, predictions))})

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(self.model.fitted_pipeline_, self.train_data["X_train"], self.train_data["y_train"], cv=5, return_times=True)

        r = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42)
        importances = r.importances_mean
        self.visualize(importances, train_sizes, train_scores, test_scores)

        mlflow.end_run()

    def visualize(self, importances, train_sizes, train_scores, test_scores):
        self.importance_plot(importances)
        self.model_performance(train_sizes, train_scores, test_scores)

    def importance_plot(self, importances):
        sorted_idx = importances.argsort()
        fig, ax = plt.subplots()
        ax.bar(range(len(sorted_idx)), importances[sorted_idx])
        ax.set_xticks(range(len(sorted_idx)))
        ax.set_xticklabels(self.features[sorted_idx], rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Permutation Importance')
        ax.set_title('Permutation Importance of Features')
        fig.tight_layout()
        plt.savefig("./figures/automl_wst_feature_30.png", dpi=300)
        
    def model_performance(self, train_size, train_scores, test_scores):
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fig, ax = plt.subplots()
        ax.grid()
        ax.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_size, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_size, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_size, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax.legend(loc="best")
        fig.savefig("./figures/automl_wst_30.png", dpi=300)