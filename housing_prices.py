import pandas as pd

# Predict the price of the SalePrice variable
filepath = './house-prices-advanced-regression-techniques/train.csv'
train_df = pd.read_csv(filepath)
print(train_df.head())

filepath2 = './house-prices-advanced-regression-techniques/test.csv'
test_df = pd.read_csv(filepath2)
print(test_df.head())
# The test does not have SalePrice and we want that.
# We wamt the average of the SalePrice
test_df['SalePrice'] = pd.Series([train_df['SalePrice'].mean()] * len(train_df))

print(test_df)

test_df[['Id', 'SalePrice']].to_csv('average_prediction.csv', index=False)
