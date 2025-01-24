from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class LogisticRegressionModel:

    def __init__(self, parameters):
        """
        Initializes the Logistic Regression.
        """
        self.model = LogisticRegression(penalty=parameters.get('penalty', 'l2'),
                              C=parameters.get('C', 1.0))

    def train(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):

        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test, metric='accuracy'):

        metric = {}
        metric['mse'] = mean_squared_error(y_test, predictions)
        metric['r2_score'] = r2_score(y_test, predictions)
        metric['mae'] = mean_absolute_error(y_test, predictions)
        metric['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))

        return metric
