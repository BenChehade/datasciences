import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

df_train = pd.read_csv('train.csv')
cols = df_train.columns
target = 'SalePrice'
variables = [column for column in df_train.columns if column!=target]

model = TSNE(n_components=2, random_state=0)
model.fit_transform(df_train)
