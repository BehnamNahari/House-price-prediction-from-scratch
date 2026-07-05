import numpy as np

from src.model import LinearRegression

def test_predict_shape():

    x = np.random.rand(10,5)

    model = LinearRegression()

    model.w = np.random.rand(5)

    pred = model.predict(x)

    assert pred.shape == (10,)

def test_fit_creates_weights():

    x = np.random.rand(100,5)

    y = np.random.rand(100)

    model = LinearRegression(
        n_epochs=10,
    )

    model.fit(
        x,
        y,
        verbose=False,
    )

    assert model.w is not None