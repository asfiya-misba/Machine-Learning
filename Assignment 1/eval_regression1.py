# Model 1
# Petal Width from Sepal Length and Sepal Width
import pickle
import numpy as np
from sklearn import datasets
from LinearRegression import LinearRegression

# Loading the model parameters
model_filepath = 'model1_with_reg.pkl'
with open(model_filepath, 'rb') as f:
    model_params = pickle.load(f)

# Creating an instance of the LinearRegression class
model = LinearRegression()
model.load('model1_with_reg.pkl')

# Setting the model parameters to the loaded values
model.weights = model_params['weight']
model.bias = model_params['bias']

# Test data
irisDataSet = datasets.load_iris()
test_X = irisDataSet.data[:, :2]  # Load or define the test input features
test_y = irisDataSet.data[:, 3:]  # Load or define the true test output values

# Predictions on test data
predictions = model.predict(test_X)

# Calculating the mean squared error value
mse = np.mean((predictions - test_y) ** 2)
print('Mean Squared Error with Regularization: ', mse)


# Loading the model parameters
model_filepath1 = 'model1_without_reg.pkl'
with open(model_filepath1, 'rb') as f:
    model_params1 = pickle.load(f)

# Creating an instance of the LinearRegression class
model2 = LinearRegression()
model2.load('model1_without_reg.pkl')

# Setting the model parameters to the loaded values
model2.weights = model_params1['weight']
model2.bias = model_params1['bias']

# Test data
irisDataSet = datasets.load_iris()
test_X = irisDataSet.data[:, :2]  # Load or define the test input features
test_y = irisDataSet.data[:, 3:]  # Load or define the true test output values

# Predictions on test data
predictions2 = model2.predict(test_X)

# Calculating the mean squared error value
mse2 = np.mean((predictions2 - test_y) ** 2)
print('Mean Squared Error without Regularization: ', mse2)


















'''

# Loading the model parameters
model_filepath = 'model1.pkl'
with open(model_filepath, 'rb') as f:
    model_params = pickle.load(f)

# Creating an instance of the LinearRegression class
model = LinearRegression()
model.load('model1.pkl')

# Setting the model parameters to the loaded values
model.weights = model_params['weight']
model.bias = model_params['bias']

# Test data
irisDataSet = datasets.load_iris()
test_X = irisDataSet.data[:, :2]  # Load or define the test input features
test_y = irisDataSet.data[:, 3:]  # Load or define the true test output values


# Predictions on test data
predictions = model.predict(test_X)

# Calculating the mean squared error value
mse = np.mean((predictions - test_y) ** 2)
print("Mean Squared Error without L2 Regularization:", mse)

modelr = LinearRegression()
modelr.load('modelr1.pkl')

# Setting the model parameters to the loaded values
modelr.weights = model_params['weight']
modelr.bias = model_params['bias']

predictions_with_reg = modelr.predict(test_X)
mse_with_reg = np.mean((predictions_with_reg - test_y) ** 2)

print("Mean Squared Error with L2 regularization:", mse_with_reg)
'''