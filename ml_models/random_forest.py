from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
class RandomForest:
    def __init__(self, parameters):
        """
        Initialize the SimpleRandomForest with a specified number of trees.
        """
        self.model = RandomForestRegressor(n_estimators=int(parameters.get('n_estimators', 100)),
                                  max_features=parameters.get('max_features', 10), random_state=int(parameters.get('random_state', 42)))
        
    def train(self, X_train, y_train):
        """
        Train the Random Forest on the provided training data.
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Predict the labels for the provided test data.
        """
        return self.model.predict(X_test)
    
    def evaluate(self, predictions, y_test):
        """
        Evaluate the accuracy of the model on the provided test data.
        """
        metric = {}
        metric['mse'] = mean_squared_error(y_test, predictions)
        metric['r2_score'] = r2_score(y_test, predictions)
        metric['mae'] = mean_absolute_error(y_test, predictions)
        metric['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))

        return metric
