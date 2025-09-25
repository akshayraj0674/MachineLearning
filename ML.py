import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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

x_test = test.copy()


numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns.tolist()


numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)


model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test_processed)

submission = sample_submission.copy()
submission['cost'] = predictions
submission.to_csv('submission.csv', index=False)