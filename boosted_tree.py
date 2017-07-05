import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, cross_val_predict,  cross_val_score
from sklearn.ensemble import gradient_boosting

df_train = pd.read_csv('train.csv')
