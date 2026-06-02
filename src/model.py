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
        n_features,n_samples = x.shape