import pandas as pd
import numpy as np
import sys
import os
#from sklearn.model_selection import train_test_split, cross_val_predict,  cross_val_score
#from sklearn.ensemble import gradient_boosting
#from sklearn.decomposition import PCA
from cat_variables import load_yml
from cat_variables import cat_var_transform
from sklearn.ensemble import RandomForestRegressor
from boruta import boruta_py
from missing_data import missing_list
from sklearn.feature_selection import SelectFromModel
import sys

#df_train = pd.read_csv('train.csv')
#df_train.drop('SalePrice', axis=1, inplace=True)
columns_to_drop = ['MasVnrArea', 'GarageYrBlt', 'LotFrontage']
#pca = PCA(n_components=30)
#pca.fit(df_train)

# combine training and test data
df_train = pd.read_csv('train.csv', keep_default_na=False, na_values=[''], index_col=0)
df_train.drop(columns_to_drop, axis=1, inplace=True)
df_train['source'] = 'train'
df_test = pd.read_csv   ('test.csv', keep_default_na=False, na_values=[''], index_col=0)
df_test.drop(columns_to_drop, axis=1, inplace=True)
df_test['source'] = 'test'
df_merge = pd.concat([df_train, df_test])

#apply changes to variables and create the train and test data
cat_dict = load_yml()
df_merge = cat_var_transform(df_train, cat_dict)
df_merge_dummy = pd.get_dummies(df_merge, columns=cat_dict['categorical'])
df_merge_dummy.replace('NA', np.nan, inplace=True)
df_train = df_merge_dummy[df_merge_dummy['source']=='train'].copy(deep=True)
df_train.drop('source', axis=1, inplace=True)
df_test = df_merge_dummy[df_merge_dummy['source']=='test'].copy(deep=True)
df_test.drop('source', axis=1, inplace=True)
df_train.to_csv('daniel.csv')

# do feature selection using boruta
X = df_train[[x for x in df_train.columns if x!='SalePrice']]
y = df_train['SalePrice']
forest = RandomForestRegressor()
feat_selector = boruta_py.BorutaPy(forest, n_estimators=1000, verbose=4)

# find all relevant features
feat_selector.fit(X.as_matrix(), y.as_matrix())

# check selected features
features_bool = np.array(feat_selector.support_)
features = np.array(X.columns)
result = features[features_bool]
print(result)

# check ranking of features
features_rank = feat_selector.ranking_
print(features_rank)
rank = features_rank[features_bool]
print(rank)




