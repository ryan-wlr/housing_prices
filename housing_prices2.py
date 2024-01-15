import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Predict the price of the SalePrice variable
filepath = './house-prices-advanced-regression-techniques/train.csv'
train_df = pd.read_csv(filepath)
print(train_df.head())

filepath2 = './house-prices-advanced-regression-techniques/test.csv'
test_df = pd.read_csv(filepath2)
print(test_df.head())

# Import two columns from the training dataset
train_df_sub = deepcopy(train_df[['MSSubClass','LotFrontage', 'SalePrice']])
print(train_df_sub)

# Mean Imputation
train_df_sub = train_df_sub.fillna(train_df_sub.mean())
print(train_df_sub)

# Make a base Linear Regression
# [col, row]
X_train, y_train = train_df_sub.to_numpy()[:,:-1], train_df_sub.to_numpy()[:, -1]
print(X_train)
print(y_train)

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
print(lr.coef_)


#https://www.youtube.com/watch?v=mSusDGZhkVU
# 22:07
print(lr.intercept_)

train_df_sub['linear_prediction'] = lr.predict(X_train)
print(train_df_sub)

mean_absolute_error(train_df_sub['SalePrice'], train_df_sub['linear_prediction'])

test_df_sub = deepcopy(test_df[['MSSubClass', 'LotFrontage']])
print(test_df_sub)

test_df_sub = test_df_sub.fillna(test_df_sub.mean())
print(test_df_sub)


X_test = test_df_sub.to_numpy()
print(X_test)

test_df['SalePrice'] = lr.predict(X_test)
print(test_df)


test_df[['Id', 'SalePrice']].to_csv('Linear_prediction.csv', index=False)

