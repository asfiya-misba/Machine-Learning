# Asfiya Misba - 1002028239
# Summer 2023
# Linear Discriminant Analysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


class LinearDiscriminantAnalysis:
    def __init__(self):
        self.mean_values = None  # To store the mean values of each class
        self.shared_covariance = None  # To store the shared covariance matrix
        self.prior_prob = None  # To store prior probabilities

    def fit(self, X, y):
        num_samples, num_features = X.shape
        classes = np.unique(y)
        num_classes = len(classes)

        self.mean_values = np.zeros((num_classes, num_features))
        self.shared_covariance = np.zeros((num_features, num_features))
        self.prior_prob = np.zeros(num_classes)

        for i, c in enumerate(classes):
            X_c = X[y == c]
            # Means represent the centroids of the data points belonging to each class
            self.mean_values[i] = np.mean(X_c, axis=0)
            self.shared_covariance += np.cov(X_c.T) * (X_c.shape[0] - 1)
            self.prior_prob[i] = X_c.shape[0] / num_samples

        self.shared_covariance /= (num_samples - num_classes)

    # Selects the class with the highest posterior probability as the predicted class label
    def predict(self, X):
        num_samples, _ = X.shape
        num_classes, _ = self.mean_values.shape

        y_pred = np.zeros(num_samples)

        for i in range(num_samples):
            posterior_prob = np.zeros(num_classes)
            for j in range(num_classes):
                delta = X[i] - self.mean_values[j]
                posterior_prob[j] = np.log(self.prior_prob[j]) - 0.5 * np.dot(
                    np.dot(delta, np.linalg.inv(self.shared_covariance)), delta)
            y_pred[i] = np.argmax(posterior_prob)

        return y_pred


# Loading the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Sepal Length and Sepal Width
y = iris.target  # Target label

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Class Means: ')
print(lda.mean_values)
print('Shared Covariance Matrix: ')
print(lda.shared_covariance)
print('Prior Probabilities: ')
print(lda.prior_prob)


# Predict the classes for the test data
y_pred = lda.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

print('Plotting the graph, please wait.')
# Plotting
plot_decision_regions(X_train, y_train, clf=lda, legend=2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1, edgecolor='k', s=100)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Linear Discriminant Analysis')
plt.show()
