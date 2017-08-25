import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
import sys
import os
#from sklearn.model_selection import train_test_split, cross_val_predict,  cross_val_score
#from sklearn.ensemble import gradient_boosting
#from sklearn.decomposition import PCA
from cat_variables import load_yml
from cat_variables import cat_var_transform
from sklearn.ensemble import RandomForestRegressor
from boruta import boruta_py
from sklearn.preprocessing import StandardScaler
from missing_data import missing_list
from sklearn.feature_selection import SelectFromModel
import sys

def generate_train(columns_to_drop = ['MasVnrArea', 'GarageYrBlt', 'LotFrontage']):
    # combine training and test data
    df = pd.read_csv('train.csv', keep_default_na=False, na_values=[''], index_col=0)
    df.drop(columns_to_drop, axis=1, inplace=True)
    df = df[df['GrLivArea']<4000]
    df['source'] = 'train'
    return df

def generate_test(columns_to_drop = ['MasVnrArea', 'GarageYrBlt', 'LotFrontage']):
    df = pd.read_csv('test.csv', keep_default_na=False, na_values=[''], index_col=0)
    df.drop(columns_to_drop, axis=1, inplace=True)
    df['source'] = 'test'
    return df

def data_merge(df_train, df_test):
    df_merge = pd.concat([df_train, df_test])

#apply changes to variables and create the train and test data
def data_manip(df):
    cat_dict = load_yml()
    df_merge = cat_var_transform(df, cat_dict)
    df_merge_dummy = pd.get_dummies(df_merge, columns=cat_dict['categorical'])
    df_merge_dummy.replace('NA', np.nan, inplace=True)
    return df_merge_dummy

    #df_train = df_merge_dummy[df_merge_dummy['source']=='train'].copy(deep=True)
    #df_train.drop('source', axis=1, inplace=True)
    #df_test = df_merge_dummy[df_merge_dummy['source']=='test'].copy(deep=True)
    #df_test.drop('source', axis=1, inplace=True)
    #df_train.to_csv('daniel.csv')

def additional_feature(df):
    #averagePrice = {2006: 153.33, 2007: 157.19, 2008: 156.30, 2009: 156.42, 2010: 157.55}
    #data about ames obtained fromhttps://fred.stlouisfed.org/series/ATNHPIUS11180Q
    #averagePrice = {2006:243066.6667,2007:243741.6667,2008:230408.3333,2009:214500,2010:221241.6667}
    #df['av_price'] = df['YrSold'].map(lambda x: averagePrice[int(x)])
    df['Bungalow'] = df['2ndFlrSF'].map(lambda x: 1*(int(x) == 0))
    df.to_csv('output3.csv')
    return df


def fs_boruta(df):
    # do feature selection using boruta
    X = df[[x for x in df.columns if x!='SalePrice']]
    y = df['SalePrice']
    forest = RandomForestRegressor()
    feat_selector = boruta_py.BorutaPy(forest, n_estimators=100, verbose=12)

    # find all relevant features
    feat_selector.fit_transform(X.as_matrix(), y.as_matrix())

    # check selected features
    features_bool = np.array(feat_selector.support_)
    features = np.array(X.columns)
    result = features[features_bool]
    #print(result)

    # check ranking of features
    features_rank = feat_selector.ranking_
    #print(features_rank)
    rank = features_rank[features_bool]
    #print(rank)

    return result

def greedy_elim(df):

    # do feature selection using boruta
    X = df[[x for x in df.columns if x!='SalePrice']]
    y = df['SalePrice']
    forest = RandomForestRegressor()
    # 150 features seems to be the best at the moment. Why this is is unclear.
    feat_selector = RFE(estimator=forest, step=1, n_features_to_select=150)

    # find all relevant features
    feat_selector.fit_transform(X.as_matrix(), y.as_matrix())

    # check selected features
    features_bool = np.array(feat_selector.support_)
    features = np.array(X.columns)
    result = features[features_bool]
    #print(result)

    # check ranking of features
    features_rank = feat_selector.ranking_
    #print(features_rank)
    rank = features_rank[features_bool]
    #print(rank)

    return result

def pca_transform(df):
    scaler = StandardScaler
    scaler.fit(df)
    scaled_data = scaler.transform(df)

def rmsle(predicted,real):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5



