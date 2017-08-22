import pandas as pd
import yaml as yml


def load_yml():
    with open('reference_file.yml', 'r') as f:
       column_dict = yml.load(f)
    return column_dict

def cat_var_transform(df, cat_dict):
    remove_items = ['categorical', 'non-categorical']
    for key in cat_dict:
        if key not in remove_items:
            print(cat_dict[key])
            df[key] = df[key].map(lambda x: cat_dict[key][x])
    return df

df_train = pd.read_csv('train.csv', keep_default_na=False)
cat_dict = load_yml()
df_train = cat_var_transform(df_train, cat_dict)
df_train = df_train[[i for i in cat_dict if i not in ['categorical', 'non-categorical']]]
df_train.to_csv('test.csv')

