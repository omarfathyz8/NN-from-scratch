from .losses import MSELoss

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = MSELoss()

    def forward(self, X):
        for layer in self.layers: X = layer.forward(X)
        return X

    def backward(self, dout):
        # assume dout is dLoss/dOutput
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if isinstance(dout, tuple):        # Dense returns (dX, dW, db)
                dout, dW, db = dout
                layer.dW, layer.db = dW, db    # cache for optimizer

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss   = self.loss_fn(y, y_pred)
        dout   = self.loss_fn.backward()
        self.backward(dout)
        return loss
