from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class SVM:
    def __init__(self, parameters):
        """
        Initializes the SVM classifier.
        
        Parameters:
        - kernel: Specifies the kernel type. Options are 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
        - C: Regularization parameter. Smaller values result in a wider margin, which might allow some misclassifications.
        """
        self.model = SVC(parameters)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        """
        Trains the SVM classifier on the provided data.
        
        Parameters:
        - X: Training data
        - y: Target labels
        """
        # Scale the data for better performance
        X = self.scaler.fit_transform(X_train)
        self.model.fit(X, y_train)

    def predict(self, X_test):
        """
        Predict the class labels for provided data.
        
        Parameters:
        - X: Data samples to predict
        
        Returns:
        - Predicted labels for each sample.
        """
        # Scale the data like we did during training
        X = self.scaler.transform(X_test)
        return self.model.predict(X_test)
    
    def evaluate(self, predictions, y_test, metric='accuracy'):
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