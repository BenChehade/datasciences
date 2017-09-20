import pandas as pd
import numpy as np
from prepare_data import data_manip, generate_train, generate_test, data_merge, fs_boruta, greedy_elim, additional_feature, rmsle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
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
bool_fs_selection = False
if bool_fs_selection:
    key_features = greedy_elim(df_train)
    key_features = [i for i in key_features]
    print(key_features)
else:
    key_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual',
                    'BsmtUnfSF', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'FireplaceQu', 'Fireplaces', 'FullBath',
                    'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GrLivArea',
                    'HalfBath', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandSlope', 'LotArea', 'LotShape', 'LowQualFinSF',
                    'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PavedDrive', 'ScreenPorch',
                    'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'Bungalow',
                    'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_80',
                    'MSSubClass_120', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl', 'Alley_NA',
                    'Alley_Pave', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl',
                    'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside',
                    'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor',
                    'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_Mitchel',
                    'Neighborhood_NAmes', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
                    'Neighborhood_OldTown', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst',
                    'Neighborhood_StoneBr','Neighborhood_Timber', 'Condition1_Artery', 'Condition1_Feedr',
                    'Condition1_Norm', 'Condition1_RRAe', 'BldgType_1Fam', 'BldgType_Duplex', 'BldgType_TwnhsE',
                    'HouseStyle_1.5Fin', 'HouseStyle_1Story', 'HouseStyle_2Story', 'HouseStyle_SLvl', 'RoofStyle_Flat',
                    'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofMatl_CompShg', 'Exterior1st_BrkFace',
                    'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood',
                    'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_Brk Cmn',
                    'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_MetalSd', 'Exterior2nd_Plywood',
                    'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng',
                    'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'Foundation_BrkTil',
                    'Foundation_CBlock', 'Foundation_PConc', 'CentralAir_N', 'CentralAir_Y', 'Electrical_FuseA',
                    'Electrical_SBrkr', 'GarageType_Attchd', 'GarageType_BuiltIn', 'GarageType_Detchd', 'GarageType_NA',
                    'MiscFeature_NA', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_NA', 'SaleType_COD','SaleType_New',
                    'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_Family', 'SaleCondition_Normal',
                    'SaleCondition_Partial']


# train and output test data
bool_split_data = True
df_train = df_train[key_features + ['SalePrice']]
df_test = df_test[key_features]
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']

# fit model 1
model_1 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.013)
#model_1 = RandomForestRegressor(n_estimators=3000, max_features=0.25)
#model_1 = Lasso(normalize=True, max_iter=100000)
#model_1 = KNeighborsRegressor()
#model_1 = SVC()

#if bool_split_data:
print(key_features)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# train model and output accuracy
df_test.fillna(0, inplace=True)
model_1.fit(X_train, y_train)
predictions = model_1.predict(X_test)
print(predictions)
df_compare = pd.DataFrame({'test': y_test, 'predictions': predictions}, index = y_test.index)
print(df_compare.head())
df_compare.to_csv('compare.csv')
sys.exit()

predictions = np.array(predictions)
y_test = np.array(y_test)
print(rmsle(predictions, y_test))
sys.exit()

# output data
predictions = pd.DataFrame(model_1.predict(df_test), index=[i for i in range(1461,2920)])
predictions.columns = ['SalePrice']
predictions.index.name = 'Id'
predictions.to_csv('boosted.csv')

















