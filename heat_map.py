# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:27:18 2017

@author: DWyatt
"""
import pandas as pd
import seaborn as sns
import sys

df_train = pd.read_csv('train.csv')
target = 'SalePrice'
variables = [column for column in df_train.columns if column!=target]

corr = df_train.corr()
sns_heat= sns.heatmap(corr, square=True)
fig = sns_heat.get_figure()
fig.savefig('heat.png')
print([target])
print(variables)
#sys.exit()

#sns_pair = sns.pairplot(df_train,
                  #x_vars=['SalePrice'],
                  #y_vars=['LotFrontage', 'Neighborhood'])
#fig = sns_pair.get_figure()
#fig.savefig('pair.png')