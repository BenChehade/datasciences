import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
################ incomplete ################

df_train = pd.read_csv('train.csv')
df_train = df_train.select_dtypes(include=[np.float])
df_train.fillna(value=0, inplace=True)
cols = df_train.columns
target = 'SalePrice'
variables = [column for column in df_train.columns if column!=target]

model = TSNE(n_components=2, random_state=0)
model.fit_transform(df_train)
