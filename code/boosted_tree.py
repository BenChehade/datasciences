import pandas as pd
import numpy as np
import sys
import os
#from sklearn.model_selection import train_test_split, cross_val_predict,  cross_val_score
#from sklearn.ensemble import gradient_boosting
#from sklearn.decomposition import PCA
from cat_variables import load_yml
from cat_variables import cat_var_transform

#df_train = pd.read_csv('train.csv')
#df_train.drop('SalePrice', axis=1, inplace=True)

#pca = PCA(n_components=30)
#pca.fit(df_train)

# apply changes to non-categorical variables
df_train = pd.read_csv('train.csv', keep_default_na=False)
cat_dict = load_yml()
df_train = cat_var_transform(df_train, cat_dict)




