# House Price Prediction from Scratch (NumPy)

A complete machine learning project implementing **Linear Regression** and **Ridge Regression** entirely from scratch using **NumPy**, without relying on machine learning libraries such as scikit-learn.

The goal of this project is to demonstrate a solid understanding of the mathematical foundations behind linear models, optimization, preprocessing, and model evaluation while following a clean and modular software engineering structure.

---

## Project Objectives

* Implement Linear Regression from scratch
* Implement Ridge (L2) Regularization
* Implement Gradient Descent optimization
* Build a modular ML pipeline
* Perform data preprocessing manually
* Evaluate model performance using multiple metrics
* Visualize model behavior and prediction quality
* Save trained models for future inference
* Write unit tests for core components

---

## Dataset

This project uses the **Ames Housing Dataset**.

The dataset contains information about residential properties in Ames, Iowa, and the task is to predict the house sale price.

Target variable:

* `SalePrice`

The target is transformed using:

```python
np.log1p(SalePrice)
```

to reduce skewness and improve optimization.

---

## Project Structure

```text
house-price-prediction/

├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── numpy_linear_baseline.ipynb
│   ├── numpy_ridge.ipynb
│   └── model_evaluation.ipynb
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   ├── model.py
│   ├── utils.py
│   └── __init__.py
│
├── tests/
│   ├── test_metrics.py
│   └── test_model.py
│
├── train.py
└── README.md
```

---

## Features

### Data Loading

* CSV loading
* Feature/target separation

### Preprocessing

* Train/validation split
* Missing value imputation (Median)
* Standardization
* Bias column generation

### Model

Implemented completely from scratch using NumPy:

* Linear Regression
* Ridge Regression
* Gradient Descent
* L2 Regularization

---

## Evaluation Metrics

The following metrics are implemented manually:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Visualizations

The evaluation notebook includes:

* Training Loss Curve
* Actual vs Predicted
* Residual Plot
* Residual Distribution
* Absolute Error Distribution

---

## Model Serialization

The trained model and preprocessing parameters are serialized using Python Pickle.

This allows the model to be reused later without retraining.

---

## Unit Testing

Basic unit tests are implemented using **pytest**.

Tests include:

* Metric correctness
* Prediction shape
* Model fitting

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Pytest

---

## Results

| Metric |   Training | Validation |
| ------ | ---------: | ---------: |
| RMSE   | **0.1497** | **0.1335** |
| R²     | **0.8619** | **0.8797** |

These results demonstrate that a simple linear model with ridge regularization can explain approximately **88%** of the variance in house prices using only numerical features.

---

## Future Improvements

Possible future extensions include:

* Feature Engineering
* Categorical Feature Encoding
* Cross Validation
* Polynomial Regression
* Advanced Tree-Based Models
* Neural Network implementation on a larger dataset

---

## Learning Outcomes

This project focuses on understanding the internal mechanics of machine learning algorithms rather than relying on high-level libraries.

Through this implementation, the following concepts were explored:

* Matrix-based Linear Regression
* Gradient Descent Optimization
* Ridge Regularization
* Feature Scaling
* Missing Value Imputation
* Modular Project Design
* Model Evaluation
* Software Engineering Practices for Machine Learning

---

## Author

Developed as part of a machine learning portfolio focused on implementing algorithms from scratch and building reproducible, well-structured ML projects.
