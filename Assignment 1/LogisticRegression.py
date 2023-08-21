# Asfiya Misba - 1002028239
# Summer 2023
# Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)


class LogisticRegression:
    def __init__(self):
        self.parameters = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        self.parameters = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        z = X.dot(self.parameters)
        return self.sigmoid(z)


def mean_squared_error(target, predicted):
    return 1 - np.square(np.subtract(target, predicted)).mean()


# Logistic Regression - Petal length/width
logreg_petal = LogisticRegression()
logreg_petal.fit(X_train[:, 2:4], y_train)
predicted_petal = logreg_petal.predict(X_test[:, 2:4])
print("Mean Squared Error - Petal length/width:", mean_squared_error(y_test, predicted_petal))

plot_decision_regions(X_train[:, 2:4], y_train, clf=logreg_petal, legend=2)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Logistic Regression - Petal length/width')
plt.show()

# Logistic Regression - Sepal length/width
logreg_sepal = LogisticRegression()
logreg_sepal.fit(X_train[:, :2], y_train)
predicted_sepal = logreg_sepal.predict(X_test[:, :2])
print("Mean Squared Error - Sepal length/width:", mean_squared_error(y_test, predicted_sepal))

plot_decision_regions(X_train[:, :2], y_train, clf=logreg_sepal, legend=2)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Logistic Regression - Sepal length/width')
plt.show()

# Logistic Regression - All features
logreg_all = LogisticRegression()
logreg_all.fit(X_train, y_train)
predicted_all = logreg_all.predict(X_test)
print("Mean Squared Error - All features:", mean_squared_error(y_test, predicted_all))

