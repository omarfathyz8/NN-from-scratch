import numpy as np

class MSELoss:
    def __call__(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        N = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / N
