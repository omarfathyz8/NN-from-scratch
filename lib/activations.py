import numpy as np
from .layers import Layer

class ReLU(Layer):
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    def backward(self, dout): return dout * self.mask

class Sigmoid(Layer):
    def forward(self, X):
        self.Y = 1 / (1 + np.exp(-X))
        return self.Y
    def backward(self, dout): return dout * self.Y * (1 - self.Y)

class Tanh(Layer):
    def forward(self, X):
        self.Y = np.tanh(X)
        return self.Y
    def backward(self, dout): return dout * (1 - self.Y ** 2)

class Softmax(Layer):
    def forward(self, X):
        # stable softmax
        X_shift = X - np.max(X, axis=1, keepdims=True)
        exp = np.exp(X_shift)
        self.Y = exp / np.sum(exp, axis=1, keepdims=True)
        return self.Y
    def backward(self, dout):
        # shortcut when softmax is paired with cross-entropy
        return dout  # caller handles full Jacobian if needed
