import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


train_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/train.csv'
test_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/test.csv'
sample_submission_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/sample_submission.csv'

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
sample_submission = pd.read_csv(sample_submission_url)


print(train.head())
print(test.head())
print(sample_submission.head())


target = 'cost'

features = [col for col in train.columns if col not in [target, 'id']]

x = train[features].values
y = train[target].values


if np.any(y <= 0):
    shift = np.abs(np.min(y)) + 1
    y_bc, bc_lambda = boxcox(y + shift)
else:
    shift = 0
    y_bc, bc_lambda = boxcox(y)


x_train, x_val, y_train_bc, y_val_bc = train_test_split(x, y_bc, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(x_train, y_train_bc)


y_val_pred_bc = lr.predict(x_val)


def inv_boxcox(y, lmbda, shift):
    if lmbda == 0:
        return np.exp(y) - shift
    else:
        return np.power(y * lmbda + 1, 1 / lmbda) - shift


y_value_pred = inv_boxcox(y_val_pred_bc, bc_lambda, shift)
y_value_true = inv_boxcox(y_val_bc, bc_lambda, shift)


lr.fit(x, y_bc)


x_test = test[features].values
y_test_pred_bc = lr.predict(x_test)
y_test_pred = inv_boxcox(y_test_pred_bc, bc_lambda, shift)


submission = sample_submission.copy()
submission[target] = y_test_pred
submission.to_csv('submission.csv', index=False)