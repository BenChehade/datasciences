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
#columns_to_drop=['Utilities']
df_train = generate_train()
df_test = generate_test()
df_merge = pd.concat([df_train, df_test])
df_merge = additional_feature(df_merge)

# create dummies and hierachical changes and split data into train and test
df_merge_transform = data_manip(df_merge)
df_train = df_merge_transform[df_merge_transform['source'] == 'train'].copy(deep=True)
df_train.drop('source', axis=1, inplace=True)
df_test = df_merge_transform[df_merge_transform['source'] == 'test'].copy(deep=True)
df_test.drop('source', axis=1, inplace=True)

# perform boruta feature selection
bool_fs_selection = False
if bool_fs_selection:
    key_features = fs_boruta(df_train)
    #key_features = greedy_elim(df_train)
    key_features = [i for i in key_features]
    print(key_features)
else:
    key_features = ['1stFlrSF', 'BsmtFinSF1', 'GarageArea', 'GrLivArea', 'LotArea', 'OverallCond', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd', 'Neighborhood_Crawfor']
    key_features1=['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual',
 'BsmtUnfSF', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'FireplaceQu', 'Fireplaces', 'FullBath', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GrLivArea',
                  'HalfBath', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandSlope', 'LotArea', 'LotFrontage', 'LotShape', 'LowQualFinSF', 'MiscVal', 'MoSold', 'OpenPorchSF',
                  'OverallCond', 'OverallQual', 'PavedDrive', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold',
                  'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl', 'Alley_Pave', 'LandContour_HLS', 'Condition1_Artery',
                  'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_RRAe', 'Condition2_Artery', 'Condition2_Norm', 'Condition2_PosN', 'LotConfig_Corner',
                  'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'Electrical_SBrkr', 'BldgType_Duplex', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr',
                  'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel',
                  'Neighborhood_NAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
                  'Neighborhood_Somerst', 'Neighborhood_StoneBr','Neighborhood_Timber', 'Neighborhood_Veenker', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin',
                  'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SLvl', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'MasVnrType_BrkCmn', 'MasVnrType_None', 'MasVnrType_Stone',
                  'Foundation_BrkTil', 'Heating_Floor', 'Heating_Grav', 'CentralAir_N', 'CentralAir_Y', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment',
                  'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'MiscFeature_None', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'Fence_GdPrv',
                  'Fence_MnPrv', 'Fence_MnWw', 'Fence_None', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New',
                  'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal',
                  'SaleCondition_Partial']

# train and output test data
bool_split_data = True
#df_train = df_train[key_features + ['SalePrice']]
#df_test = df_test[key_features]
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']

# fit model 1
#model_1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.25, max_depth=2, min_samples_split=3)
model_1 = GradientBoostingRegressor(n_estimators=2500, learning_rate=0.04, max_depth=3, min_samples_split=2)
#model_1 = RandomForestRegressor(n_estimators=3000, max_features=0.25)
#model_1 = Lasso(normalize=True, max_iter=100000)
#model_1 = KNeighborsRegressor()
#model_1 = SVC()

#if bool_split_data:
print(key_features)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=84)

# train model and output accuracy
model_1.fit(X_train, y_train)
predictions = model_1.predict(X_test)
df_compare = pd.DataFrame({'test': y_test, 'predictions': predictions}, index = y_test.index)
print(df_compare.head())
df_compare.to_csv('compare.csv')

predictions = np.array(predictions)
y_test = np.array(y_test)
print(rmsle(predictions, y_test))
sys.exit()
# output data
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
model_2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.04, max_depth=5, min_samples_split=3)
model_2.fit(X,y)
predictions = pd.DataFrame(model_2.predict(df_test), index=[i for i in range(1461,2920)])
predictions.columns = ['SalePrice']
predictions.index.name = 'Id'
predictions.to_csv('boosted.csv')

















