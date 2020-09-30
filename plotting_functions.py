# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:38:47 2020

@author: georgems
"""
import pandas as pd
import numpy as np
import dateutil

from matplotlib import pyplot as plt
import seaborn
seaborn.set()


def calculate_fig_dim(diag_size, resolution):
    """
    Calculate the width/height and dpi of the plot required for the given
    screen dimensions. Input diag_size in inches and resolution as a list.
    """
    ratio = resolution[0] / resolution[1]
    
    height = diag_size / np.sqrt(1 + ratio**2)
    width = ratio * height
    
    dpi = resolution[1] / height
    
    return [width, height], dpi


def plot_wallpaper_line(data, figsize, dpi, facecolor, linecolor):
    # Set figure/axes
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)

    ax = plt.axes()
    ax.set_facecolor(facecolor)  # dark deep blue background

    data.plot(color=linecolor)

    # Set glow effect by plotting multiple overlapping lines with low alpha
    # values
    n_lines = 10
    diff_linewidth = 1.05
    alpha_value = 0.03

    for n in range(1, n_lines + 1):
        data.plot(linewidth=2 + (diff_linewidth * n),
                  alpha=alpha_value,
                  color=linecolor)

    plt.axis('off')

    # Adjust axis limits so that the line occupies the middle 1/3 horizontally
    height = max(data) - min(data)
    plt.ylim(min(data) - height, max(data) + height)
    
    return fig, ax


# ---- Labels
def annotate_period(ax, label_str, start_date, end_date):
    """
    Add a |-| label and text beneath the plot to mark a time period.

    Parameters
    ----------
    ax: axes
        The axes to apply the label to
    label_str : str
        The text to display beneath the |-| line.
    start_date : datetime
        The start date for the |-| line.
    end_date : datetime
        The end date for the |-| line.

    Returns
    -------
    ax: axes

    """
    mid_date = start_date + (end_date - start_date) / 2
    
    # Add text
    ax.annotate(label_str, xy=(mid_date, -60), xytext=(0, -20),
                xycoords='data', ha='center', textcoords='offset points',
                color='white')
    
    # Add line
    ax.annotate('', xy=(start_date, -100), xytext=(end_date, -100),
                xycoords='data', textcoords='data',
                arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })

    return ax


def annotate_date(ax, data,
                  label_str, date,
                  top_correction_factor=0,
                  bottom_correction_factor=0,
                  pos='top'):
    """
    Add a vertical dashed line above the plot, with a text annotation rotated
    90 degrees such that it is underlined by the line.

    Parameters
    ----------
    ax: axes
        The axes to apply the label to
    data: pd.Series
        The data used to plot the main line
    label_str : str
        The text to display on the label.
    date : datetime
        The date that the line points to on the plot.
    [top|bottom]_correction_factor: float
        The change in height of the top or bottom of the plot line, given as
        a ratio of the original height.
    pos : str, optional
        Position for the text. The default is 'top'.

    Returns
    -------
    ax: axes

    """
    # Get the word count for the given date
    y_value = data.loc[date]
    
    bottom_of_line = y_value * (1 + bottom_correction_factor)
    
    graph_range = max(data) - min(data)
    top_of_line = y_value + graph_range * 2 / 3
    top_of_line = top_of_line * (1 + top_correction_factor)
    
    # Add the label rotated 90 degrees
    ax.annotate(label_str,
                xy=(date, top_of_line),
                xytext=(-10, 0),
                xycoords='data',
                va=pos,
                textcoords='offset points',
                rotation=90,
                color='white',
                size='x-small')
    
    # Add vertical dashed line
    ax.annotate('',
                xy=(date, bottom_of_line),
                xytext=(date, top_of_line),
                xycoords='data',
                textcoords='data',
                arrowprops={'arrowstyle': '-', 'ls': 'dashed', 'lw': 1})
    
    return ax
