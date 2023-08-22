import numpy as np # for number operations

class LinearRegression:
    def __init__(self, lr = 0.001, epochs = 1000):
        self.weights = None
        self.bias = None
        self.epochs = epochs # Number of iterations
        self.lr = lr

    def fit(self, X, y):
        n_samples,n_features = X.shape

        self.bias = 0
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = np.dot(X,self.weights) + self.bias

            delta_weights = (-2/n_samples) * np.dot(X.T, (y-y_pred))
            delta_bias = (-2/n_samples) * np.sum(y-y_pred)

            self.bias = self.bias - self.lr * delta_bias # new values of bias/b
            self.weights = self.weights - self.lr * delta_weights # new values of slope/weights/m

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias