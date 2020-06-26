# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:53:07 2020

Program to calculate useful statistics from messenger chat histories, 
particularly from group chats (more than two people), and then plot them.

To obtain the required DataFrame run load_and_preprocess.py.

@author: georgems
"""

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Set working directory
os.chdir(r'C:/Users/georg/Documents/Python/Messenger')

# Load the main DataFrame from file
df = pickle.load( open("gc_df.pkl", "rb") )


#%% ==== SUMMARISE DATA ====
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
    
    stat_df.loc["total_angry", name] = count_spec_word(temp[temp.notnull()].raw, "hello") 
    
    # Reactions given by "name"
    temp = df[("reac", name)]
    stat_df.loc["total_reac_by"] = temp[temp.notnull()].shape[0]
    stat_df.loc["top_reac", name] = top_k_words(temp[temp.notnull()], k=7)
    
    stat_df.loc["total_heart", name] = count_spec_word(temp[temp.notnull()], "😍")
    stat_df.loc["total_angry", name] = count_spec_word(temp[temp.notnull()], "😠")
    stat_df.loc["total_wow", name] = count_spec_word(temp[temp.notnull()], "😮")
    stat_df.loc["total_cry", name] = count_spec_word(temp[temp.notnull()], "😢")
    stat_df.loc["total_up", name] = count_spec_word(temp[temp.notnull()], "👍")
    stat_df.loc["total_down", name] = count_spec_word(temp[temp.notnull()], "👎")
    
    
    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(stat_df.T[["total_messages_sent", "total_words_sent", "total_emojis_sent"]])


# Transpose and reset index for plotting purposes
stat_df_t = stat_df.T.reset_index()
stat_df_t.rename(columns = {"index":"name"}, inplace=True)


# Create a dataframe of reaction information
#   > rows are the sender of the message
#   > columns are the giver of the reaction
#   > entries are total reactions given to [row name] by [col name]

reac_df = pd.DataFrame(index = participant_names, columns = participant_names,
                       dtype="float")

for name_on in participant_names:
    for name_by in participant_names:
        reac_df.loc[name_on, name_by] = df[df.sender_name == name_on][("reac", name_by)].count()


#%% ==== DISTRIBUTION PLOTS ====
sns.catplot("sender_name","word_count", data=df, kind="strip", jitter=0.4, aspect=2.75)
plt.xticks(rotation=90)
plt.xlabel("Participant")
plt.ylabel("Word count per message")
plt.title("Distribution of word count across messages")
plt.ylim(0,200)

sns.catplot("sender_name","emo_count", data=df, kind="strip", jitter=0.4, aspect=2.75)
plt.xticks(rotation=90)
plt.xlabel("Participant")
plt.ylabel("Emoji count per message")
plt.title("Distribution of emoji count across messages")
plt.ylim(0,50)


#%% ==== REACTION PLOTS ====
plot_df = stat_df_t[["name", "total_heart", "total_angry", "total_wow", "total_cry", "total_up", "total_down"]]
plot_df = pd.melt(plot_df, id_vars="name", var_name="reac", value_name="count")

sns.catplot(x="name", y="count", data=plot_df, hue="reac", kind="bar", aspect=2.5)
plt.xticks(rotation=90)
plt.title("Number of reactions by each person")

sns.catplot(x="reac", y="count", data=plot_df, hue="name", kind="bar", aspect=2.5)
plt.xticks(rotation=90)
plt.title("Number of reactions by each person")


#%% ==== MESSAGE COUNT PLOT AGAINST TIME ====
df["message_count"] = np.repeat(1, df.shape[0])

by_day = df.pivot(columns = "sender_name", values = "message_count")
by_day = by_day.resample("M").sum()
by_day = pd.melt(by_day.reset_index(), id_vars="timestamp_ms", value_vars=by_day.columns, 
                 var_name="sender_name", value_name="message_count")


fig = plt.figure(figsize=[12,9], dpi=157)
sns.lineplot(x="timestamp_ms", y="message_count", data=by_day, hue="sender_name")
plt.xlabel("Date")
plt.ylabel("Number of messages")
plt.title("Number of messages sent over time (grouped by month)")


#%% ==== REACTION HEATMAP
fig = plt.figure(figsize=[12,9], dpi=157)

sns.heatmap(data = reac_df, linewidths=0.5)
plt.xlabel("Reaction-giver")
plt.ylabel("Reaction-receiver")
plt.title("Heatmap of who reacts to who")