import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer


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


imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)


model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_leaf=200,
    min_samples_split=500,
    random_state=0
)
model.fit(x_train_imputed, y_train)


predictions = model.predict(x_test_imputed)


submission = sample_submission.copy()
submission['cost'] = predictions
submission.to_csv('submission.csv', index=False)