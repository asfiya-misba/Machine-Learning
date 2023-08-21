# Asfiya Misba - 1002028239
# Summer 2023
# Linear Regression
import pickle
import numpy as np


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.weights = None
        self.alpha = 0.001  # Learning rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.bias = None
        self.error = []
        self.steps = []
        self.l2_error = []
        self.l2_loss = []
        self.mse_loss = []

    # To calculate gradient
    def l2_gradient(self, X, y):
        error_term = ((np.dot(X, self.weights) + self.bias) - y)
        dot_product = np.dot(X.T, error_term)
        regularized_term = (self.regularization * self.weights)
        return (1 / len(y)) * dot_product + regularized_term

    # To calculate L2 regularization
    def l2_regularization(self, X, y, reg_lambda_term):
        squared_error = np.sum(((np.dot(X, self.weights) + self.bias) - y) ** 2)
        normal_regularization_term = np.mean(squared_error)
        return normal_regularization_term + reg_lambda_term

    # To perform gradient descent with L2 regularization
    def gradient_descent_l2(self, X, y, batch_size, alpha, regularization):
        self.regularization = regularization
        total_batches = X.shape[0] // batch_size
        loss = []

        for i in range(total_batches + 1):
            start = i * batch_size
            end = (i + 1) * batch_size
            input_x = X[start:end, :]
            target = y[start:end]
            output = input_x @ self.weights + self.bias
            sum_of_square_of_weight = np.sum(np.square(self.weights))
            lambda_2m = (self.regularization / (2 * batch_size))
            reg_lambda_term = lambda_2m * sum_of_square_of_weight
            self.l2_loss.append(self.l2_regularization(input_x, target, reg_lambda_term))
            # Calculating the derivatives
            weight_derivative = self.weights - alpha * self.l2_gradient(input_x, target)
            bias_derivative = self.bias - alpha * ((1 / len(output)) * np.sum(output - target))
            self.weights = weight_derivative
            self.bias = bias_derivative
            loss.append(self.l2_regularization(input_x, target, reg_lambda_term))
        # For remaining data points
        if X.shape[0] % batch_size != 0:
            start = total_batches * batch_size
            input_x = X[start:X.shape[0]]
            target = y[start:X.shape[0]]
            output = input_x @ self.weights + self.bias
            sum_of_square_of_weight = np.sum(np.square(self.weights))
            lambda_2m = (self.regularization / (2 * batch_size))
            lambda_reg = lambda_2m * sum_of_square_of_weight
            self.l2_loss.append(self.l2_regularization(input_x, target, lambda_reg))
            weight_derivative = self.weights - alpha * self.l2_gradient(input_x, target)
            bias_derivative = self.bias - alpha * ((1 / len(output)) * np.sum(output - target))
            self.weights = weight_derivative
            self.bias = bias_derivative
            loss.append(self.l2_regularization(input_x, target, lambda_reg))
        return np.mean(loss)

    # To train the linear regression model
    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        alpha = 0.01  # Learning rate
        num_samples, num_features = X.shape
        # Random normalized weight
        self.weights = np.random.randn(num_features, y.shape[1])
        # Random normalized bias
        self.bias = np.random.randn(1, y.shape[1])
        start = int(0.9 * num_samples)
        val_X = X[start:]
        val_y = y[start:]
        X = X[:start]
        y = y[:start]
        best_weights = None
        best_bias = None
        best_loss_value = float('inf')
        better_val = 0
        for epoch in range(max_epochs):
            self.steps.append(epoch)
            self.l2_error.append(self.gradient_descent_l2(X, y, batch_size, alpha, regularization))
            val_pred = val_X @ self.weights + self.bias
            lambda_2m = (self.regularization / (2 * batch_size))
            sum_of_square_of_weight = np.sum(np.square(self.weights))
            lambda_reg = lambda_2m * sum_of_square_of_weight
            loss_value = lambda_reg + np.mean(np.square(val_pred - val_y))
            if loss_value < best_loss_value:
                best_loss_value = loss_value
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                better_val = 0
            else:
                better_val += 1
                if better_val == patience:
                    break
        self.weights = best_weights
        self.bias = best_bias

    # To predict the target values
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    # To calculate the MSE value
    def score(self, X, y):
        prediction = self.predict(X)
        squared_error = np.sum((prediction - y) ** 2)
        meanSquaredError = squared_error / (len(X) * len(y))
        return meanSquaredError

    # To save the model
    def save(self, filepath):
        model_params = {'weight': self.weights, 'bias': self.bias}
        with open(filepath, 'wb') as f:
            pickle.dump(model_params, f)

    # To load the model
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_params = pickle.load(f)
        self.weights = model_params['weight']
        self.bias = model_params['bias']
