# Model 3
# Sepal Length from Petal Length and Petal Width
import pickle
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt
from LinearRegression import LinearRegression

irisDataSet = datasets.load_iris()
x = irisDataSet.data[:, 2:4]  # Petal Length and Petal Width
y = irisDataSet.data[:, :1]  # Sepal Length
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=7, stratify=irisDataSet.target)

# Train the model without regularization
model3_without_reg = LinearRegression()
model3_without_reg.fit(X_train, y_train, 32, 0, 100, 3)

# Train the model with regularization
model3_with_reg = LinearRegression()
model3_with_reg.fit(X_train, y_train, 32, regularization=0.1, max_epochs=100, patience=3)

# Saving the models
model3_without_reg.save('model3_without_reg.pkl')
model3_with_reg.save('model3_with_reg.pkl')

print("Best Weight Value: ", model3_without_reg.weights)
print("Best Bias Value: ", model3_without_reg.bias)

plt.plot(model3_without_reg.steps, model3_without_reg.l2_error)
plt.title('Predicting Sepal Length from Petal Length and Petal Width')
plt.xlabel('Steps')
plt.ylabel('MSE Loss')
plt.show()
