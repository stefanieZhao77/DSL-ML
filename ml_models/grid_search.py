from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import mlflow
class GridSearch:

    def __init__(self, name, model, param_grid, task, cv=5):
        self.model = model
        self.param_grid = self.handle_params(param_grid)
        self.task = task
        self.cv = cv
        self.model_name = name
        self.best_model = None
        self.best_params = None

    def handle_params(self, params):
        new_params = {}
        for param in params.keys():
            new_params[param] = []  
            if param == "n_estimators" or param == "random_state":
                new_params[param].extend([int(i) for i in params[param]])
            else:
                new_params[param].extend(i for i in params[param])  
        return new_params
    
    def train(self, X_train, y_train):
        if(self.task == 'regression'):
            if self.model == 'SVM':
                self.model = GridSearchCV(SVR(), self.param_grid, cv=self.cv)
            elif self.model == 'DecisionTree':
                self.model = GridSearchCV(DecisionTreeRegressor(), self.param_grid, cv=self.cv)
            elif self.model == 'RandomForest':
                self.model = GridSearchCV(RandomForestRegressor(), self.param_grid, cv=self.cv)
        elif(self.task == 'classification'):
            if self.model == 'SVM':
                self.model = GridSearchCV(SVC(), self.param_grid, cv=self.cv)
            elif self.model == 'DecisionTree':
                self.model = GridSearchCV(DecisionTreeClassifier(), self.param_grid, cv=self.cv)
            elif self.model == 'RandomForest':
                self.model = GridSearchCV(RandomForestClassifier(), self.param_grid, cv=self.cv)
        

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        self.best_model = self.model.best_estimator_
        self.best_params = self.model.best_params_
        mlflow.log_params(self.best_params)
        mlflow.sklearn.log_model(self.best_model, "best_" + self.model_name)
        return self.best_model.predict(X_test)

    def evaluate(self, predictions, y_test):      
        metric = {}
        metric['mse'] = mean_squared_error(y_test, predictions)
        metric['r2_score'] = r2_score(y_test, predictions)
        metric['mae'] = mean_absolute_error(y_test, predictions)
        metric['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        return metric 
