# Model 1
# Petal Width from Sepal Length and Sepal Width
import pickle
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt
from LinearRegression import LinearRegression

irisDataSet = datasets.load_iris()
x = irisDataSet.data[:, :2]  # Sepal Length and Sepal Width
y = irisDataSet.data[:, 3:]  # Petal Width
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=7, stratify=irisDataSet.target)

# Train the model without regularization
model1_without_reg = LinearRegression()
model1_without_reg.fit(X_train, y_train, 32, 0, 100, 3)

# Train the model with regularization
model1_with_reg = LinearRegression()
model1_with_reg.fit(X_train, y_train, 32, regularization=0.1, max_epochs=100, patience=3)

# Saving the models
model1_without_reg.save('model1_without_reg.pkl')
model1_with_reg.save('model1_with_reg.pkl')

print("Best Weight Value: ", model1_without_reg.weights)
print("Best Bias Value: ", model1_without_reg.bias)

plt.plot(model1_without_reg.steps, model1_without_reg.l2_error)
plt.title('Predicting Petal Width from Sepal Length and Sepal Width')
plt.xlabel('Steps')
plt.ylabel('MSE Loss')
plt.show()