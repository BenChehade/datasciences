import pandas as pd
import numpy as np
from prepare_data import data_manip, generate_train, generate_test, data_merge, fs_boruta, greedy_elim, additional_feature, rmsle
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import sys

# generate test and training data and merge them together for manipulation (dummies and hierachical)
df_train = generate_train()
df_test = generate_test()
df_merge = pd.concat([df_train, df_test])
df_merge = additional_feature(df_merge)

# create dummies and hierachical changes and split data into train and test
df_merge_transform = data_manip(df_merge)
df_train = df_merge_transform[df_merge_transform['source'] == 'train'].copy(deep=True)
df_train.drop('source', axis=1, inplace=True)
df_train= df_train.apply(pd.to_numeric)
df_test = df_merge_transform[df_merge_transform['source'] == 'test'].copy(deep=True)
df_test.drop('source', axis=1, inplace=True)
df_test= df_test.apply(pd.to_numeric)

# perform boruta feature selection
bool_fs_selection = True
if bool_fs_selection:
    key_features = greedy_elim(df_train)
    key_features = [i for i in key_features]
    print(key_features)
else:
    key_features = [i for i in df_test.columns]
    #key_features = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtQual', 'GarageArea', 'GarageCars',
 #'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF',
 #'YearBuilt', 'YearRemodAdd']

# train and output test data
bool_split_data = False
df_train = df_train[key_features + ['SalePrice']]
df_test = df_test[key_features]
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
forest = RandomForestRegressor()
if bool_split_data:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    forest.fit(X_train,y_train)
    predictions = forest.predict(X_test)
    predictions = np.array(predictions)
    y_test = np.array(y_test)
    print(rmsle(predictions, y_test))

else:
    forest.fit(X,y)
    df_test.fillna(0, inplace=True)
    predictions = pd.DataFrame(forest.predict(df_test), index=[i for i in range(1461,2920)])
    predictions.columns = ['SalePrice']
    predictions.index.name = 'Id'
    predictions.to_csv('output.csv')

















