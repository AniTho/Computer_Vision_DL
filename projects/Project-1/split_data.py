import pandas as pd
import pathlib
import os

def extract_fname(row):
    path_str = str(row.path)
    fname = path_str[path_str.rindex('/')+1:]
    return fname

def extract_age(row):
    stop_idx = row.fname.index('_')
    age = int(row.fname[:stop_idx])
    return age

def extract_gender(row):
    start_idx = row.fname.index('_')
    gender_idx = int(row.fname[start_idx+1:start_idx+2])
    if gender_idx:
        return 'female'
    return 'male'

def split(img_dir, train_split = 0.8, valid_split = 0.2):
    if os.path.exists('data/UTKFace/data.csv'):
        print("File already exists")
        return
    img_dir = pathlib.Path(img_dir)
    images_list = list(img_dir.glob('*.jpg'))
    df = pd.DataFrame(images_list, index = list(range(len(images_list))), columns=['path'])
    df['fname'] = df.apply(lambda row: extract_fname(row), axis = 1)
    df['age'] = df.apply(lambda row: extract_age(row), axis = 1)
    df['gender'] = df.apply(lambda row: extract_gender(row), axis = 1)
    df = df.sample(frac = 1, random_state = 42).reset_index(drop=True)
    df['split_type'] = None
    train_columns = int(train_split*len(df))
    valid_columns = int(valid_split*len(df))
    train_idxs = list(range(train_columns))
    valid_idxs = list(range(train_columns, train_columns + valid_columns))
    test_idxs = list(range(train_columns + valid_columns, len(df)))
    df.loc[train_idxs, 'split_type'] = 'train' 
    df.loc[valid_idxs, 'split_type'] = 'valid'
    df.loc[test_idxs, 'split_type'] = 'test'
    df.drop(['path'], axis = 1, inplace=True)
    df.to_csv('data/UTKFace/data.csv', index=False)

    