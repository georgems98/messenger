# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:59:15 2020

Program to perform to perform the initial processing of the original JSON 
files downloaded from Facebook.

The end result is a Pandas DataFrame where 
    > the rows are individual messages sent in the chat
    > the columns are 
        >> name of the sender of the message
        >> message content
        >> reactions given to the message by each chat participant
        >> five versions of the message content
        >> count of the number of raw words
    > the index is the message timestamp

@author: georgems
"""

import os
import json
import string
import emoji
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np


# Set working directory 
os.chdir(r'C:/Users/georg/Documents/Python/Messenger')


#%% ==== LOAD ORIGINAL JSON FILES ====

def pd_load_messages(loc_str):
    """
    Load the original downloaded JSON files, extract the message information, 
    and return a dataframe.
    N.B. the JSON files has message info (sender name, message content, etc)
    in the "messages" key.
    
    Parameters
    ----------
    loc_str : str
        A string giving the directory which contains the JSON files.

    Returns
    -------
    DataFrame
        A DataFrame indexed by timestamp where each entry is a message.
    """
    
    df = pd.DataFrame()
        
    # For each JSON file, load it and add the messages to the DataFrame
    for f in os.listdir(loc_str):
        file = loc_str + "/" + f
        data = json.load(open(file, 'r'))
        df = df.append(pd.DataFrame(data['messages']), ignore_index=True)
      
    # Keep only the meaningful information
    df = df[['sender_name', 'content', 'reactions', 'photos', 'timestamp_ms']]
    
    # The data is encoded as latin-1 so must be recoded as utf8 to preserve 
    # emojis
    def code(content):
        try:
            return content.encode('latin1').decode('utf8')
        except AttributeError:
            pass
            
    # Recode message content
    df['content'] = df.loc[:,'content'].apply(code)

    # TODO
    # Keep multilevel column name indexing - seems to disappear after line 90

    # Create new multilevel col with colnames (new_reactions, name)
    participant_names = list(set(df.sender_name))
    multi_cols = pd.MultiIndex.from_product([["reac"], participant_names])
    reac_col = pd.DataFrame(np.nan, index=df.index, columns = multi_cols)
    reac_count_col = pd.DataFrame(np.nan, index=df.index, columns=["reac_count"])
    
    df = pd.concat([df, reac_col, reac_count_col], axis=1)
    
    for index, row in df[df.reactions.notnull()].iterrows():
        df.loc[index, "reac_count"] = len(row.reactions)
        
        for dic in row.reactions:
            df.loc[index, ("reac", dic["actor"])] = code(dic["reaction"])
        
    
    # Convert the POSIX format to Timestamp and then index DataFrame by it
    df['timestamp_ms'] = pd.to_datetime(df['timestamp_ms']/1000, unit='s')
    df = df.set_index("timestamp_ms")
    
    return df
                

df = pd_load_messages('gc_message_data')
df.head()


#%% ==== PROCESS MESSAGE CONTENT ====
# Input a string, output five versions of it as a tuple of lists
def preprocess_text(content_str):
    """
    Process a string of text to extract the following versions of it:
        > raw
        > stop words removed
        > stemmed
        > lemmatised
        > emojis only
    
    Parameters
    ----------
    content_str : str
        A string containing the content to be processed.
    
    Returns
    -------
    Five lists
        Each version of the text is returned as a list whose elements are 
        individual 'words'.

    """
    
    # Function to convert a list of lists into a single list by joining them 
    # end-to-end in the obvious way
    def flatten_list(two_list):
        one_list = []
        
        for el in two_list:
          one_list.extend(el)
          
        return one_list
    
    # A translation table that doesn't replace anything but does 
    # remove any punctuation (string.punctuation is a list of punctuation)
    # i.e. str.maketrans replaces the string '' with '', and replaces 
    # string.punctuation with None
    translator = str.maketrans('', '', string.punctuation)
    
    # Create a list of the sentences
    upd = nltk.sent_tokenize(str(content_str))
    
    
    # Remove punctuation
    lines = [line.translate(translator) for line in upd]
    # Tokenize as words
    lines = [nltk.word_tokenize(line) for line in lines]
    # Lowercase
    lines = [ [word.lower() for word in line if word not in ['\'', '’', '”', '“'] ] for line in lines ]
    
    # Raw data with no punctuation
    raw = lines
    
    # Remove stop words
    stop = [[word for word in line if word not in set(stopwords.words('english'))] for line in raw]
    
    # Find stems
    snowball_stemmer = SnowballStemmer('english')
    stem = [[snowball_stemmer.stem(word) for word in line] for line in stop]
    
    # Find lemmas
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma = [ [wordnet_lemmatizer.lemmatize(word) for word in line] for line in stop ]

    # Find emoji
    emo = [ [char for char in lem if char in emoji.UNICODE_EMOJI] for lem in lemma]


    raw = flatten_list(raw)
    stop = flatten_list(stop)
    stem = flatten_list(stem)
    lemma = flatten_list(lemma)
    
    # Find emoji - NB working with flattened lemma
    emo = [ [char for char in lem if char in emoji.UNICODE_EMOJI] for lem in lemma]
    
    emo = flatten_list(emo)

    return raw, stop, stem, lemma, emo


# Call function on the 'content' col and create a new column that contains 
# the output tuple, then move each of the outputs to its own column and delete
# the original col
df.loc[:,'processed'] = df.loc[:,'content'].apply(preprocess_text)
df[['raw','stop','stem','lemma','emo']] = pd.DataFrame(df.loc[:,'processed'].tolist(), index = df.index)
df = df.drop('processed', axis = 1)

# Count the number of words in the raw list of each message
df['word_count'] = df.loc[:,'raw'].apply(lambda x: len(x))
df['emo_count'] = df.loc[:,'emo'].apply(lambda x: len(x))


pickle.dump(df, open("gc_df.pkl", 'wb'))
