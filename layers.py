import numpy as np

class Layer:
    def forward(self, X): raise NotImplementedError
    def backward(self, dout): raise NotImplementedError

class Dense(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        self.X, self.Z = None, None          # cache for backward

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self, dout):
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dX = dout @ self.W.T
        return dX, dW, db
