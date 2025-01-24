from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class DecisionTree:

    def __init__(self, parameters):
        self.model = DecisionTreeRegressor(max_depth= int(parameters.get('max_depth', 10)), min_samples_split=int(parameters.get('min_samples_split', 2)), min_samples_leaf=int(parameters.get('min_samples_leaf', 1)))

    def train(self, X_train, y_train):
        """
        Trains the decision tree classifier.

        The function splits the dataset into training and testing sets, then fits the classifier on the training data.
        It then makes predictions on the test data and prints the accuracy of the classifier.
        """

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict the class labels for provided data.

        Parameters:
        - X: Data samples to predict

        Returns:
        - Predicted labels for each sample.
        """
        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test):
        """
        Evaluates the accuracy of the classifier.

        Parameters:
        - predictions: Predicted labels for each sample.
        - y_test: True labels for each sample.

        Returns:
        - Accuracy of the classifier.

        """
        metric = {}
        metric['mse'] = mean_squared_error(y_test, predictions)
        metric['r2_score'] = r2_score(y_test, predictions)
        metric['mae'] = mean_absolute_error(y_test, predictions)
        metric['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        return metric