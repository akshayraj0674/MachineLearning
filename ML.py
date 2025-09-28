import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


train_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/train.csv'
test_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/test.csv'
sample_submission_url = 'https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/project-1-me-4127-e-2025-26/sample_submission.csv'

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
sample_submission = pd.read_csv(sample_submission_url)


print(train.head())
print(test.head())
print(sample_submission.head())


target_col = 'cost'
id_col = 'id'

features_cols = [col for col in train.columns if col not in [target_col, id_col]]


x = train[features_cols]
y = train[target_col]
x_test = test[features_cols]


x = x.fillna(x.median())
x_test = x_test.fillna(x_test.median())


scaler = StandardScaler()
x = scaler.fit_transform(x)
x_test = scaler.transform(x_test)


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


model = HuberRegressor()
model.fit(x_train, y_train)


val_pred = model.predict(x_val)

test_pred = model.predict(x_test)


submission = sample_submission.copy()
submission[target_col] = test_pred
submission.to_csv('submission.csv', index=False)