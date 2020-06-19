# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:29:10 2020

Program to generate polar plots of chat data.
N.B. run load_and_preprocess.py first to obtain required DataFrame 

Theta axis can be in minutes, hours, days of the week, or months of the year.
R/modulus axis is word count, broken down by chat participant.


@author: georgems
"""

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set working directory
os.chdir(r'C:/Users/georg/Documents/Python/Messenger')

# Load the main DataFrame from file
df = pickle.load( open("main_df.pkl", "rb") )


#%% ==== POLAR PLOTS ====

def get_plot_coords(df, freq, melt=True):
    """
    Get the coordinates needed for a polar plot of time data by resampling and 
    calculating the theta and r values

    Parameters
    ----------
    df : DataFrame
        Main DataFrame of data.
    freq : str
        Resampling frequency ("T", "H", "D", "M").
    melt : bool, optional
        Return a melted DataFrame if True. The default is True.

    Returns
    -------
    df : DataFrame
        DataFrame ready for plotting.

    """
    # Save the participant names for later
    names = list(set(df.sender_name))
    
    # Pivot such that the columns are the sender names and the values are word 
    # counts, then resample
    df = df.pivot(columns = "sender_name", values = "word_count")
    df = df.resample(freq).sum()
    
    # Access the appropriate property for grouping based on freq
    d = {"T": df.index.time, "H": df.index.time, "D": df.index.dayofweek, "M": df.index.month}
    df = df.groupby(d[freq]).sum()
    df["time"] = df.index
    
    # Calculate the theta values
    length = df.shape[0]
    df["angle"] = np.pi/2 - np.linspace(0, length-1, length)*2*np.pi/length
    
    # Add the first value to the end so that graph is joined at the ends
    df = df.append(df.iloc[0]) 
    
    if melt is True:
        # Melt to tidy form (unpivot)
        df = pd.melt(df, id_vars="angle", value_vars=names, 
                     var_name="name", value_name="word_count")
    
    return df
    
    
def polar_plot(df, title, ticks, labels, 
               legend = None, shade = True, height = 8):
    """
    Polar plot for time grouped data.

    Parameters
    ----------
    df : DataFrame
        Main DataFrame of data
    ticks : list
        Position of the theta ticks in radians ([pi/2, -3*pi/2]).
    labels : list
        The text to display next to the theta ticks.
    legend : tuple, optional
        Tuple of strings of the legend text. The default is None, in which
        case the names from the DataFrame are used.
    shade : bool, optional
        If true then fill in the area inside the curves. The default is True.
    height : float, optional
        Size of plot. The default is 8.

    Returns
    -------
    None.

    """
        
    polar = sns.FacetGrid(df, height = height, hue = "name",
                        subplot_kws=dict(projection="polar"),
                        sharex=True, sharey=True, despine=False)
    polar.map(plt.plot, "angle", "word_count")
    
    polar.set(xticks = ticks, xlim = [np.pi/2, -3*np.pi/2], 
              xticklabels = labels,
              title = title,
              xlabel = "", ylabel="")
    
    if legend is not None:
        plt.legend(legend, loc="best")
    else:
        plt.legend(list(set(df.name)), loc="best")
    
    if shade is True:
        for name in set(df.name):
            plt.fill_between(df[df.name==name].angle,
                             df[df.name==name].word_count, 
                             alpha=0.2)
    

sns.set_style("whitegrid")
#sns.set_palette("Blues_r")

# ---- Plot against hour of the day
by_time = get_plot_coords(df, freq='H')
ticks = np.linspace(np.pi/2,-3*np.pi/2,24, endpoint=False)
labels = [str(int(x)) for x in np.linspace(0,23,24)]
leg = ("George", "Sarah")
polar_plot(by_time, "Word count by hour",ticks, labels, leg)

# ---- Plot against day of the week
by_day = get_plot_coords(df, freq = 'D')
ticks = np.linspace(np.pi/2, -3*np.pi/2, 7, endpoint=False)
labels = ["Mon", "Tues", "Weds", "Thurs", "Fri", "Sat", "Sun"]
polar_plot(by_day, "Word count by day", ticks, labels, leg)

# ---- Plot against month
by_month = get_plot_coords(df, freq = 'M')
ticks = np.linspace(np.pi/2, -3*np.pi/2, 12, endpoint=False)
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
polar_plot(by_month, "Word count by month", ticks, labels, leg)
