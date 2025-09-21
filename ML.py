import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


train_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/train.csv'
test_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/test.csv'
sample_submission_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/sample_submission.csv'

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
sample_submission = pd.read_csv(sample_submission_url)


print(train.head())
print(test.head())
print(sample_submission.head())


x_train = train.drop('cost', axis=1)
y_train = train['cost']

x_test = test


model = LinearRegression()
model.fit(x_train, y_train)


predictions = model.predict(x_test)