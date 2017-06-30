# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:27:18 2017

@author: DWyatt
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

df_train = pd.read_csv('train.csv')
cols = df_train.columns
target = 'SalePrice'
variables = [column for column in df_train.columns if column!=target]

k = 10
corr = df_train.corr()
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
sns_heat = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                       xticklabels=cols.values)
plt.yticks(rotation=0)
plt.xticks(rotation='vertical')
plt.tight_layout()
fig = sns_heat.get_figure()
fig.savefig('topTen - heatMap.png')
print([target])
print(variables)
#sys.exit()

#sns_pair = sns.pairplot(df_train,
                  #x_vars=['SalePrice'],
                  #y_vars=['LotFrontage', 'Neighborhood'])
#fig = sns_pair.get_figure()
#fig.savefig('pair.png')