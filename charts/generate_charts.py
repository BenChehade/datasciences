# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:43:46 2017

@author: DWyatt
"""

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
target = 'SalePrice'
variables = [str(column) for column in df_train.columns if column!=target]
dir1 = os.getcwd()
for variable in variables:
    plt.figure()
    sns_scatter= sns.stripplot(x=variable, y=target, data=df_train)
    fig = sns_scatter.get_figure()
    fig.savefig(dir1 + '\\charts\\' + variable + '.png')
    del sns_scatter
    del fig
    #del df_train

#sys.exit()

#sns_pair = sns.pairplot(df_train,
                  #x_vars=['SalePrice'],
                  #y_vars=['LotFrontage', 'Neighborhood'])