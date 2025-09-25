import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


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


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


model = SVR(kernel='rbf', C=1.0, gamma='scale')
model.fit(x_train, y_train)


y_val_pred = model.predict(x_val)
mse = mean_squared_error(y_val, y_val_pred)
print("Validation Mean Squared Error:", mse)


test_pred = model.predict(test)


submission = sample_submission.copy()
submission['cost'] = test_pred
submission.to_csv('submission.csv', index=False)