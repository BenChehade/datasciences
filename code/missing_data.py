import pandas as pd

def missing_list(df):
    #df_train = pd.read_csv('train.csv')
    #df_train.drop('SalePrice', axis=1, inplace=True)
    #df_test = pd.read_csv('test.csv')
    #df = pd.concat([df_train, df_test], axis=0, keys=['train', 'test'])
    #df.to_csv('missing.csv')
    remove_list = []
    missing = (df.isnull().sum().sort_values(ascending=False)/(df.shape[0]))*100
    missing = missing[missing>0]
    remove_list+=[i for i in missing.index[missing>15]]
    print(missing)
    print(remove_list)
df_train = pd.read_csv('train.csv')
df_train.drop('SalePrice', axis=1, inplace=True)
df_test = pd.read_csv('test.csv')
df = pd.concat([df_train, df_test], axis=0, keys=['train', 'test'])
missing_list(df)