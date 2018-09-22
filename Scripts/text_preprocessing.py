import pandas as pd
import numpy as np

import time
from tqdm import tqdm
import datetime as dt

import re
from  nltk.stem.snowball import SnowballStemmer


def df_processing(df):
    """
    Main function to remove special characters and extract hashtags, parties, mentions from a dataframe
    Takes dataframe that has been imported using data handling script. 
    -----
    Necessary column: 'full_text'
    -----
    Creates new columns: 
    clean_text: based on full text, does not contain mentions, retweets, umlauts and hyperlinks are replaced.
    tags: hashtags from full text
    mentions: mentions (@) from full text
    parties: all parties mentioned in full text
    stemmed: based on clean text, uses the nltk snowball stemmer to stem words
    """
    
    df['tags'] = df['full_text'].map(tag_finder)
    df['mentions'] = df['full_text'].map(at_finder)
    df['clean_text'] = df['full_text'].map(link_finder).\
                                       map(at_remover).\
                                       map(text_cleaner).\
                                       map(rt_remover).\
                                       map(replace_german_umlaut).\
                                       map(number_remover)
    df['parties'] = df['full_text'].map(party_finder)
    df['stemmed'] = df['clean_text'].map(stemmi)



"""
Functions that cleans the text:
- remove special characters, extract mentions, tags and parties
- remove retweets
- replace umlaute
"""

# removes special characters from text - # and @ to be removed together with the subsequent word in separate function
text_cleaner = lambda x: " ".join(re.findall(r"[A-Za-z0-9üäöÜÄÖß@]*", x))

# remove numbers from text
number_remover = lambda x: re.sub(r'(\d)+', '', x)

# extracts hashtags+subsequent word from full_text
tag_finder = lambda x: re.findall(r"#(\w+)", x)

# removes hashtags+subsequent word from full_text
# tag_remover = lambda x: re.sub("#(\w+)", '', x)

# extracts mentions(@) and subsequent word
at_finder = lambda x: re.findall(r"@(\w+)", x)

# removes mentions(@) and subsequent word
at_remover = lambda x: re.sub(r"@(\w+)", '', x)

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
    
    parties = ['afd', 'spd', 'cdu', 'csu', 'fdp', 'linke', 'gruene']
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
        df['clean_text'] = df['full_text'].apply(text_cleaner).map(rt_remover).astype('str')
        df['parties'] = df['clean_text'].map(party_finder)
               
def stemmi(x):
    """
    Function that iterates over a given column and returns stemmed words
    """
    return ' '.join([SnowballStemmer('german').stem(y) for y in x.split(' ')])        


def replace_german_umlaut(unicode_string):
    """
    Function to replace German umlauts with more international characters - predominantly removes the problem with the Gruenen
    """

    umlaute_dict = {b'\xc3\xa4': b'ae',     
                b'\xc3\xb6': b'oe', 
                b'\xc3\xbc': b'ue', 
                b'\xc3\x84': b'Ae', 
                b'\xc3\x96': b'Oe',  
                b'\xc3\x9c': b'Ue',  
                b'\xc3\x9f': b'ss'}

    utf8_string = unicode_string.encode('utf-8')

    for k in umlaute_dict.keys():
        utf8_string = utf8_string.replace(k, umlaute_dict[k])

    return utf8_string.decode()