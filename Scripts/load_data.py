import os
import pandas as pd
import numpy as np

import time
from tqdm import tqdm
import datetime as dt
import pickle

def foo():
    print('die zweite2')

def read_data(file = 'json_file_location.json'):
    """
    Function to load single json files
    """
    
    data_frame = pd.read_json(file, 
                               orient='records', 
                               lines=True, 
                               convert_dates=True, 
                               keep_default_dates=True)
    return data_frame



def cleaning(df):
    """
    Creates a dataframe with distinct columns only and only german tweets; 
    defines a datetime index and sorts in an ascending manner; 
    also, deletes duplicates in the text column
    """
    
    
    ger_df = df.loc[:, ['created_at', 'id_str', 'full_text']][df.lang == 'de']
    ger_df = ger_df.set_index(pd.DatetimeIndex(ger_df['created_at'], inplace=True)).sort_values('created_at', ascending=True)
    
    clean_ger_df = ger_df.drop_duplicates('full_text')

    assert len(clean_ger_df) == (len(ger_df) - ger_df.duplicated('full_text').sum())
    
    return ger_df


def load_data(file_list):
    """
    Function that loads json files from a list of files, using the read_data function,
    and appends the resulting pandas DataFrames to each other
    """
    
    df = pd.DataFrame()
    
    for file in tqdm(file_list):
        if os.path.isfile(file):
            sub_df = cleaning(read_data(file))
        else:
            print('File {} not found'.format(file))
            continue
        if df.empty:
            df = sub_df
        else:
            df = df.append(sub_df)
            
    return df