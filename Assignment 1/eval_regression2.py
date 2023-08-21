# Model 2
# Sepal Width from Petal Length and Petal Width
import pickle
import numpy as np
from sklearn import datasets
from LinearRegression import LinearRegression

# Loading the model parameters
model_filepath = 'model2_with_reg.pkl'
with open(model_filepath, 'rb') as f:
    model_params = pickle.load(f)

# Creating an instance of the LinearRegression class
model = LinearRegression()
model.load('model2_with_reg.pkl')

# Setting the model parameters to the loaded values
model.weights = model_params['weight']
model.bias = model_params['bias']

# Test data
irisDataSet = datasets.load_iris()
test_X = irisDataSet.data[:, 2:4]  # Load or define the test input features
test_y = irisDataSet.data[:, 1:2]  # Load or define the true test output values

# Predictions on test data
predictions = model.predict(test_X)

# Calculating the mean squared error value
mse = np.mean((predictions - test_y) ** 2)
print('Mean Squared Error with Regularization: ', mse)


# Loading the model parameters
model_filepath1 = 'model2_without_reg.pkl'
with open(model_filepath1, 'rb') as f:
    model_params1 = pickle.load(f)

# Creating an instance of the LinearRegression class
model2 = LinearRegression()
model2.load('model2_without_reg.pkl')

# Setting the model parameters to the loaded values
model2.weights = model_params1['weight']
model2.bias = model_params1['bias']

# Test data
irisDataSet = datasets.load_iris()
test_X = irisDataSet.data[:, 2:4]  # Load or define the test input features
test_y = irisDataSet.data[:, 1:2]  # Load or define the true test output values

# Predictions on test data
predictions2 = model2.predict(test_X)

# Calculating the mean squared error value
mse2 = np.mean((predictions2 - test_y) ** 2)
print('Mean Squared Error without Regularization: ', mse2)
