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
            df[key] = df[key].map(lambda x: cat_dict[key][x])
    return df

