# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:53:07 2020

Program to calculate useful statistics from messenger chat histories, 
particularly from group chats (more than two people). 

To obtain the required DataFrame run load_and_preprocess.py.

@author: georgems
"""

import os
import pickle
from collections import Counter

import pandas as pd


# Set working directory
os.chdir(r'C:/Users/georg/Documents/Python/Messenger')

# Load the main DataFrame from file
df = pickle.load( open("main_df.pkl", "rb") )


#%%

def top_k_words(series, k=10):
    """
    Function that returns the k most common words contained in the inputted 
    column. E.g. supply df.raw as "series".
    """
    full_count = Counter()

    for row in series:
        for el in row:
            if el != "none":         # A photo message can have content "none"
                full_count[el] += 1
 
    return full_count.most_common(k)


def count_spec_word(series, word):
    """
    Function that returns the number of times "word" is contained in the 
    inputted column. E.g. supply df.raw as "series".
    """
    
    full_count = Counter()
    
    for row in series:
       for el in row:
           if el == word.lower():
               full_count[el] += 1

    return full_count[word.lower()]


# Initialise DataFrame with columns as each chat participant
participant_names = list(set(df.sender_name))

rows = ["total_messages_sent", "total_words_sent", "total_emojis_sent", 
        "total_photos_sent", "top_words", "top_emojis", "max_reac_received", 
        "total_reac_received", "total_reac_by", "top_reac"]

stat_df = pd.DataFrame(index=rows, columns=participant_names)

# Fill in data for each of the requested categories
for name in stat_df.columns:
    # Messages sent by "name"
    temp = df[df.sender_name == name]
    
    stat_df.loc["total_messages_sent", name] = temp.shape[0]
    stat_df.loc["total_words_sent", name] = sum(temp.word_count)
    stat_df.loc["total_emojis_sent", name] = sum(temp.emo_count)
    stat_df.loc["total_photos_sent", name] = sum(temp.photos.notnull())
    stat_df.loc["top_words", name] = top_k_words(temp[temp.lemma.notnull()].lemma, k=10)
    stat_df.loc["top_emojis", name] = top_k_words(temp[temp.emo.notnull()].emo, k=10)
    stat_df.loc["max_reac_received", name] = temp.reac_count.max()
    stat_df.loc["total_reac_received", name] = sum(temp.reac_count.fillna(0))
    stat_df.loc["hello_count", name] = count_spec_word(temp[temp.notnull()].raw, "hello") 
    
    # Reactions given by "name"
    temp = df[("reac", name)]
    stat_df.loc["total_reac_by"] = temp[temp.notnull()].shape[0]
    stat_df.loc["top_reac", name] = top_k_words(temp[temp.notnull()], k=7)
    

