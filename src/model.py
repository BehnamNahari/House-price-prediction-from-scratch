import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01,n_epochs=1000,reg_lambda=0.01):
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg_lambda = reg_lambda
        self.w = None
        self.loss_history = []

    def _initialize_weights(self,n_features):
        rng = np.random.RandomState(42)
        self.w = rng.normal(loc=0.0, scale=0.01, size=n_features)

    def predict(self,x):
        return x @ self.w

    def compute_loss(self, x, y):
        n = x.shape[0]
        y_pred = self.predict(x)
        mse = np.mean((y_pred - y)**2)
        reg = self.reg_lambda * np.sum(self.w[1:] ** 2)
        return mse + reg

    def fit(self,x,y,x_val=None,y_val=None,verbose=True):
        n_samples, n_features = x.shape
        self._initialize_weights(n_features)

        for epoch in range(self.n_epochs):
            y_pred = self.predict(x)
            errors = y_pred - y
            grad_w = (2/n_samples) * (x.T @ errors)
            grad_w[1:] += 2 * self.reg_lambda * self.w[1:]
            self.w -= self.lr * grad_w
            train_loss = self.compute_loss(x, y)
            self.loss_history.append(train_loss)
            if verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1 ):
                msg = f'Epoch {epoch:4d} | train_loss = {train_loss:.6f}'
                if x_val is not None and y_val is not None:
                    val_loss = self.compute_loss(x_val, y_val)
                    msg += f' | val_loss = {val_loss:.6f}'
                    print(msg)
        return self