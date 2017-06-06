"""
Just some stuff
"""
import pandas as pd
fname='train.csv'
data = pd.read_csv(fname)
print(data.describe())