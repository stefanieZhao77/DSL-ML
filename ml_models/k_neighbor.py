from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class KNeighborModel():

    def __init__(self, parameters):
        self.model = KNeighborsRegressor(n_neighbors=int(parameters.get('n_neighbors', 5)), weights=parameters.get('weights', 'uniform'), algorithm=parameters.get('algorithm', 'auto'))

    def train(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):

        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test):

        metric = {}
        metric['mse'] = mean_squared_error(y_test, predictions)
        metric['r2_score'] = r2_score(y_test, predictions)
        metric['mae'] = mean_absolute_error(y_test, predictions)
        metric['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))

        return metric
    
