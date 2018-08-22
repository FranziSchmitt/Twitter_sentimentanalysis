import os
import pandas as pd
import numpy as np

import time
from tqdm import tqdm
import datetime as dt
import re




# removes special characters from text - # and @ to be removed together with the subsequent word in separate function
text_cleaner = lambda x: " ".join(re.findall(r"[A-Za-z0-9üäöÜÄÖß@]*", x))

# extracts hashtags+subsequent word from full_text
tag_finder = lambda x: re.findall(r"#(\w+)", x)

# removes hashtags+subsequent word from full_text
# tag_remover = lambda x: re.sub("#(\w+)", '', x)

# extracts mentions(@) and subsequent word
at_finder = lambda x: re.findall("@(\w+)", x)

# removes mentions(@) and subsequent word
at_remover = lambda x: re.sub("@(\w+)", '', x)

# removes retweets(RT) from clean text
rt_remover = lambda x: re.sub(r"RT ", '', x)

# substitute links with the string 'link'
link_finder = lambda x: re.sub(r"(http|https)://[\w\-]+(\.[\w\-]+)+\S*", 'einLink', x)

# finds all parties mentioned in text after hashtags are removed
def party_finder(x):
    parties = ['AfD', 'SPD', 'CDU', 'CSU', 'FDP', 'LINKE', 'Gruene', 'Grüne']
    text = x.lower()
    out = []
    for party in parties:
        if party.lower() in text:
            out.append(party)
    return out

def party_remover(x):
    """
    removes party mentions and tags from column
    """
    
    parties = ['afd', 'spd', 'cdu', 'csu', 'fdp', 'linke', 'gruene', 'grüne']
    text = x.lower()
    clean = str()
    
    for party in parties:
        if party in text:
            clean = text.replace(party, '')
    return clean

def dict_processing(dict_of_DFs, parties):
    """
    Function to remove special characters and extract hashtags, parties, mentions from a dictionary of dataframes
    """
    
    for party in parties:
        df = dict_of_DFs[party]
        df['tags'] = df['full_text'].apply(tag_finder)
        df['clean_text'] = df['full_text'].apply(text_cleaner).apply(tag_remover).map(rt_remover).astype('str')
        df['parties'] = df['clean_text'].map(party_finder)
        
        
        
def df_processing(df):
    """
    Function to remove special characters and extract hashtags, parties, mentions from a dataframe
    """
    
    df['tags'] = df['full_text'].apply(tag_finder)
    df['mentions'] = df['full_text'].apply(at_finder)
    df['clean_text'] = df['full_text'].map(link_finder).apply(text_cleaner).apply(at_remover).map(rt_remover)
    df['parties'] = df['clean_text'].map(party_finder)