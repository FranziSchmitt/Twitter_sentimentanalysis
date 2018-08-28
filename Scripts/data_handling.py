import os
import pandas as pd
import numpy as np

import time
from tqdm import tqdm
import datetime as dt
import pickle


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
    
    
    ger_df = df.loc[:, ['created_at', 'full_text', 'user']][df.lang == 'de']
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


def collapse_dfs(df_dict):
    """
    Function that 
    1. takes a dictionary of dataframes, iterates over them and creates a single dataframe, containing all data
    2. deduplicates the dataframe
    """

    parties = ['AfD', 'SPD', 'CDU', 'CSU', 'FDP', 'LINKE', 'Gruene', 'Grüne']
    all_data = pd.DataFrame()

    for party in parties:
        df = df_dict[party]
        all_data = all_data.append(df)
  
    
    return all_data

def load_large(file_list):
    """
    Function that loads more than just a minimal dataframe used for clustering or classification.
    Variation of the load_data function.
    """

    df = pd.DataFrame()
    
    for file in tqdm(file_list):
        if os.path.isfile(file):
            sub_df = selecting(read_data(file))
        else:
            print('File {} not found'.format(file))
            continue
        if df.empty:
            df = sub_df
        else:
            df = df.append(sub_df)
            
    return df

def selecting(df):
    large_df = df.loc[:, :][df.lang == 'de']
    large_df = large_df.set_index(pd.DatetimeIndex(large_df['created_at'], inplace=True)).sort_values('created_at', ascending=True)
    
    clean_large_df = large_df.drop_duplicates('full_text')

    assert len(clean_large_df) == (len(large_df) - large_df.duplicated('full_text').sum())
    
    return large_df