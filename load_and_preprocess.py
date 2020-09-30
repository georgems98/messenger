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
import pandas as pd
import numpy as np
import typer

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import logging
FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def load_json(directory):
    """
    Returns a dataframe of message data
    """
    LOGGER.info(f"Loading data from {directory}")
    
    full_dir = os.listdir(directory)
    full_dir = [os.path.join(directory, x)
                for x in full_dir
                if x.endswith(".json")]
    n = len(full_dir)
    LOGGER.critical(f"Found {n} JSON files")
    
    df_list = [None] * n
    
    for i in range(n):
        filepath = full_dir[i]
        data = json.load(open(filepath, 'r'))
        df_list[i] = pd.DataFrame(data['messages'])
    
    return pd.concat(df_list, ignore_index=True)


def recode(content):
    """
    The data is encoded as latin-1 so must be recoded as utf8 to preserve
    emojis.
    """
    try:
        return content.encode('latin1').decode('utf8')
    except AttributeError:
        pass
    
    
def parse_reactions(df):
    """
    Convert reactions from series of lists of dictionaries to multilevel col
    and recode emojis.
    """
    LOGGER.info("Parsing reactions")
    participant_names = list(set(df.sender_name))
    df.columns = pd.MultiIndex.from_product([df.columns, [""]])

    # For reactions use multiindex columns with names as sublevel
    # All others have original colname as main level
    for name in participant_names:
        df["reac", name] = np.nan
    
    df["reac_count"] = df.loc[df["reactions"].notnull(),
                              "reactions"].apply(len)
    df["reac_count"] = df["reac_count"].fillna(0)
    
    reactions = df.loc[df["reactions"].notnull(), "reactions"].explode()
    reactions = pd.DataFrame(reactions.tolist(), index=reactions.index)
    reactions.columns = pd.MultiIndex.from_product([reactions.columns, [""]])
    
    df = df.join(reactions)
    
    for name in participant_names:
        mask = df["actor"] == name
        df["reac", name] = df.loc[mask, "reaction"]
    
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(["actor", "reaction", "reactions"], axis=1)
    
    for name in participant_names:
        mask = df["reac", name].notnull()
        df.loc[mask, ("reac", name)] = df.loc[mask,
                                              ("reac", name)].apply(recode)
    
    return df


def parse_timestamp(df):
    """
    Convert the POSIX format to datetime.
    """
    df['timestamp'] = (pd.to_datetime(df['timestamp_ms'] / 1000, unit='s')
                       .dt
                       .to_pydatetime())

    return df


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
    upd = nltk.sent_tokenize(content_str)
    
    # Remove punctuation, tokenise to words, and lowercaseize
    lines = [line.translate(translator) for line in upd]
    lines = [nltk.word_tokenize(line) for line in lines]
    lines = [[word.lower()
              for word in line
              if word not in ['\'', '’', '”', '“']]
             for line in lines]
    
    # Raw data with no punctuation
    raw = lines
    
    # Remove stop words
    stop = [[word
             for word in line
             if word not in set(stopwords.words('english'))]
            for line in raw]
    
    # Find stems
    snowball_stemmer = SnowballStemmer('english')
    stem = [[snowball_stemmer.stem(word) for word in line] for line in stop]
    
    # Find lemmas
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma = [[wordnet_lemmatizer.lemmatize(word)
              for word in line]
             for line in stop]

    # Find emoji
    emo = [[char
            for char in lem
            if char in emoji.UNICODE_EMOJI]
           for lem in lemma]


    raw = flatten_list(raw)
    stop = flatten_list(stop)
    stem = flatten_list(stem)
    lemma = flatten_list(lemma)
    
    # Find emoji - NB working with flattened lemma
    emo = [[char
            for char in lem
            if char in emoji.UNICODE_EMOJI]
           for lem in lemma]
    
    emo = flatten_list(emo)

    return raw, stop, stem, lemma, emo


def parse_content(df):
    """
    Apply the preprocessing function to give 5 new columns and 
    raw/emo word counts
    """
    LOGGER.info("Parsing message content")
    # Call function on the 'content' col and create a new column that contains
    # the output tuple, then move each of the outputs to its own column and
    # delete the original col
    mask = df["content"].notnull()

    result = df.loc[mask, "content"].apply(preprocess_text)
    result = pd.DataFrame(result.tolist(),
                          columns=['raw', 'stop', 'stem', 'lemma', 'emo'],
                          index=df[mask].index)

    result.columns = pd.MultiIndex.from_product([["content"], result.columns])
    
    df = df.join(result)
    
    # Count the number of words in the raw list of each message
    df["word_count"] = df.loc[df["content", "raw"].notnull(), ("content", "raw")].apply(len)
    df["emo_count"] = df.loc[df["content", "emo"].notnull(), ("content", "emo")].apply(len)
    df["word_count"] = df.loc[df["word_count"].notnull(), "word_count"].fillna(0)
    df["emo_count"] = df.loc[df["emo_count"].notnull(), "emo_count"].fillna(0)
    
    return df


def main(
    input_dir: str = typer.Option("raw_data",
                                  help="The directory to search for JSON messenger data"),
    output_dir: str = typer.Option("parsed_data",
                                   help="The directory to output to"),
):
    """
    Load the original downloaded JSON files, extract and parse the message
    information, and return a dataframe.
    N.B. the JSON files has message info (sender name, message content, etc)
    in the "messages" key.
    """
    
    df = load_json(input_dir)
    LOGGER.critical(f"Initial dataframe size: {df.shape}")
    
    # Keep only the meaningful information
    df = df[['sender_name', 'content', 'reactions', 'photos', 'timestamp_ms']]
    
    df["content"] = df["content"].apply(recode)
    
    df = parse_reactions(df)
    df = parse_timestamp(df)
    df = parse_content(df)
    
    df = df.set_index("timestamp")
    
    as_of_date = max(df.index).strftime('%Y%m%d')
    filepath = os.path.join(output_dir, f"message_data_{as_of_date}.pkl")
    
    LOGGER.info(f"Writing to {filepath}")
    pickle.dump(df, open(filepath, 'wb'))
    
    LOGGER.info("Program complete")


if __name__ == "__main__":
    typer.run(main)
