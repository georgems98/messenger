# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:35:16 2020

Program to generate an aesthetic plot of messenger chat data (not for 
generating "useful" data visuals).

The plot has date on the x-axis and total word count from the chat history on 
the y-axis. Annotations and date period labels can be added above and below 
the main line respectively.

The appropriate size and resolution of the output image can be set from the 
dimensions of the display (i.e. the diagonal size in inches and resolution).


@author: George Molina-Stubbs
"""

# Standard library
import os
import datetime
import math
import pickle

# Anaconda
from matplotlib import pyplot as plt
import seaborn; seaborn.set()
import pandas as pd


os.chdir(r'C:/Users/georg/Documents/Python/Messenger')

# Load the DataFrame and resample by calendar day, dropping zero entries
df = pickle.load( open("main_df.pkl", "rb") )
df_rs = df.resample('D').sum()
df_rs = df_rs[df_rs.word_count != 0]
df_rs = df_rs.word_count


#%% ==== PLOT ====
# Pull out the datetime and count data from the DataFrame
x = df_rs.index
y = df_rs.T.values


def fig_dim(diag_size, resolution):
    """
    Calculate the width/height and dpi of the plot required for the given 
    screen dimensions. Input diag_size in inches and resolution as a list.
    """
    ratio = resolution[0]/resolution[1]
    
    height = diag_size/math.sqrt(1 + ratio**2)
    width = ratio*height
    
    dpi = resolution[1]/height
    
    return [width, height], dpi


# ---- Main plot 
# Plot size and dpi for 28inch monitor in 4K
size, dpi = fig_dim(28, [3840,2160])

# Set figure/axes
fig = plt.figure(figsize = size, dpi=dpi, facecolor='#061f3e') # Set size
ax = plt.axes()

ax.set_facecolor('#061f3e') # dark deep blue background

plt.plot(x, y, color = '#FFCD33') # gold main line

# Set glow effect by plotting multiple overlapping lines with low alpha values
n_lines = 10
diff_linewidth = 1.05
alpha_value = 0.03

for n in range(1, n_lines+1):    
    plt.plot(x,y, linewidth=2+(diff_linewidth*n), 
              alpha=alpha_value,
              color = '#FFCD33') # gold

plt.axis('off')

# Adjust axis limits so that the line occupies the middle 1/3 horizontally
height = max(y) - min(y)
plt.ylim(min(y)-height, max(y)+height)


# ---- Labels

# Function to annotate time periods
# Input dates as a string in the format '2019.1.12'
def annotate_period(label_str, start_str, end_str):
    """
    Add a |-| label and text beneath the plot to mark a time period.

    Parameters
    ----------
    label_str : str
        The text to display beneath the |-| line.
    start_str : str
        The start date for the |-| line.
    end_str : str
        The end date for the |-| line.

    Returns
    -------
    None.

    """
    start_split = start_str.split('.')
    start_date = datetime.datetime(int(start_split[0]), int(start_split[1]), int(start_split[2]))
    
    end_split = end_str.split('.')
    end_date = datetime.datetime(int(end_split[0]), int(end_split[1]), int(end_split[2]))
    
    mid_date = start_date + (end_date - start_date)/2
    
    # Add text
    ax.annotate(label_str, xy=(mid_date, -60), xycoords='data', ha='center', 
                xytext=(0, -20), textcoords='offset points', color='white')
    
    # Add line
    ax.annotate('', xy=(start_date, -100), xytext=(end_date, -100), 
                xycoords='data', textcoords='data', 
                arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })


# Function to annotate specific dates
def annotate_date(label_str, date_str, bottom = 0, height = 2000, pos = 'top'):
    """
    Add a vertical dashed line above the plot, with a text annotation rotated
    90 degrees such that it is underlined by the line.
    Note the max height of the line is 6000.

    Parameters
    ----------
    label_str : str
        The text to display on the label.
    date_str : str
        The date that the line points to on the plot.
    bottom : int, optional
        Y-location for the bottom of the line (added to the y value of the 
        main plot line). The default is 0.
    height : int, optional
        Y-location for the top of the line (added to the y value of the 
        main plot line). The default is 2000.
    pos : str, optional
        Position for the text. The default is 'top'.

    Returns
    -------
    None.

    """
    # Get datetime format for date_str
    date_date = pd.to_datetime(date_str)
    
    # Get the word count for the given date
    y_value = df_rs.loc[date_date] 
    
    # Add the label rotated 90 degrees
    ax.annotate(label_str, xy=(date_date, min((y_value + height), 6000)), 
                xycoords='data', va=pos, xytext=(-10, 0), 
                textcoords='offset points', rotation=90, 
                color='white', size='x-small')
    
    # Add vertical dashed line
    ax.annotate('', xy=(date_date, y_value + bottom), 
                xytext=(date_date, min((y_value + height), 6000)), 
                xycoords='data', textcoords='data', 
                arrowprops={'arrowstyle': '-', 'ls': 'dashed', 'lw': 1})


# Add dates to the plot
annotate_date('5km run', '2018.11.19')
annotate_date('Gatwick drone', '2018.12.19', height = 3000)
annotate_date('Phone call', '2018.12.24', height = 3600)
annotate_date('Browns', '2019.5.6')
annotate_date('Trinity may ball', '2019.6.17')
annotate_date('Cava on Erasmus', '2019.6.24')
annotate_date('Newcastle', '2019.8.22')
annotate_date('Afternoon tea', '2019.12.8', height = 3000)
annotate_date('Diss', '2019.12.21')
annotate_date('Italian restaurant', '2020.2.8', height = 2500)
annotate_date('Reunion', '2020.5.11', height = 3000)


# Add time periods to the plot
annotate_period('Summer 2018', '2018.7.6', '2018.9.29')
annotate_period('Christmas 2018', '2018.12.1', '2019.1.12')
annotate_period('Easter 2019', '2019.3.15', '2019.4.20')
annotate_period('Switzerland', '2019.6.30', '2019.7.28')
annotate_period('Christmas 2019', '2019.12.6', '2020.1.11')
annotate_period('Lockdown', '2020.3.23', '2020.5.18')


# Manually input this one to highlight specific visits
ax.annotate('Summer visits', xy=(datetime.datetime(2019,8  ,30), -60), xycoords='data', ha='center', xytext=(0, -20), textcoords='offset points', color='white')
ax.annotate('', xy=(datetime.datetime(2019,8,8), -100), xytext=(datetime.datetime(2019,8,15), -100), 
            xycoords='data', textcoords='data',     
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
ax.annotate('', xy=(datetime.datetime(2019,8,22), -100), xytext=(datetime.datetime(2019,8,27), -100), 
            xycoords='data', textcoords='data',     
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
ax.annotate('', xy=(datetime.datetime(2019,9,5), -100), xytext=(datetime.datetime(2019,9,9), -100), 
            xycoords='data', textcoords='data',     
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
ax.annotate('', xy=(datetime.datetime(2019,9,13), -100), xytext=(datetime.datetime(2019,9,21), -100), 
            xycoords='data', textcoords='data',     
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })


# Done!
plt.show()
