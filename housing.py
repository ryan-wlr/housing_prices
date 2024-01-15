import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuild', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train.head()

from sklearn.ensemble import RandomForrestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimator=100, random_state=0)
model_3 = RandomForestRegressor(n_estimate=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimate=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimate=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

    for i in range(0, len(models)):
        mae= score_model(models[i])
        print("Model %d MAE: %d" % (i+1, mae))

# Fill in best model
best_model = model_3

# Define a model
my_model = RandomForestRegressor()

# Fit the model to the training data
my_model.fit(X,y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index, 'SalePrice':preds_test})

output.to_csv('submission.csv', index=False)
