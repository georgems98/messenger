# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:49:40 2020

@author: georgems
"""

import pandas as pd
import numpy as np
import pickle
import typer
import dateutil

from matplotlib import pyplot as plt

from plotting_functions import (
    calculate_fig_dim,
    plot_wallpaper_line,
    annotate_date,
    annotate_period,
    )

import logging
FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def load_data(filepath, colname):
    LOGGER.info(f"Loading parsed data from {filepath}")
    df = pickle.load(open(filepath, "rb"))
    data = df[colname].resample('D').sum()
    # data = data[data != 0]
    
    return data


def load_labels(filepath):
    LOGGER.info(f"Loading label data from {filepath}")
    labels = pd.read_csv(filepath)
    
    labels["date1"] = labels["date1"].apply(dateutil.parser.parse)
    labels["date2"] = labels.loc[labels["date2"].notnull(),
                                 "date2"].apply(dateutil.parser.parse)
    
    labels = labels.replace({pd.NaT: np.nan})
    
    labels["top_correction_factor"] = labels["top_correction_factor"].fillna(0)
    labels["bottom_correction_factor"] = labels["bottom_correction_factor"].fillna(0)
    
    return labels


def verify_dates(labels, data):
    """
    Verify the label dataframe against the available lineplot data
    Remove entries with invalid dates
    """
    min_date = min(data.index)
    max_date = max(data.index)
    
    mask = ((min_date > labels.loc[labels["date1"].notnull(), "date1"])
            | (max_date < labels.loc[labels["date1"].notnull(), "date1"])
            | (min_date > labels.loc[labels["date2"].notnull(), "date2"])
            | (max_date < labels.loc[labels["date2"].notnull(), "date2"]))

    if not labels[mask].empty:
        LOGGER.warning(f"Invalid dates found: {labels[mask]}")
        return labels[~mask]
    else:
        LOGGER.info("All dates valid")
        return labels


def main(
        input_file: str = typer.Option("parsed_data/message_data_20200914.pkl",
                                       help="Location of pickled data"),
        label_file: str = typer.Option("wallpaper_labels.csv",
                                       help="Location of the csv of label data"),
        output_file: str = typer.Option("word_count_wallpaper.png",
                                        help="Location to output wallpaper to"),
        plot_kind: str = typer.Option("word_count",
                                      help="The name of the column to take as the y-values in the plot"),
        screen_size: float = typer.Option(28,
                                          help="Diagonal screen size in inches"),
        screen_resolution: str = typer.Option("3840x2160",
                                              help="Screen resolution, e.g. `3840x2160`"),
):
    data = load_data(input_file, plot_kind)
    
    horizontal = int(screen_resolution.split("x")[0])
    vertical = int(screen_resolution.split("x")[1])
    figsize, dpi = calculate_fig_dim(screen_size, [horizontal, vertical])

    facecolor = "#061f3e"   # dark blue
    linecolor = "#FFCD33"   # gold

    fig, ax = plot_wallpaper_line(data, figsize, dpi, facecolor, linecolor)

    labels = load_labels(label_file)
    labels = verify_dates(labels, data)
    
    for indx, row in labels[labels["date2"].isnull()].iterrows():
        annotate_date(ax,
                      data,
                      row["label"],
                      row["date1"],
                      row["top_correction_factor"],
                      row["bottom_correction_factor"])
    
    for indx, row in labels[labels["date2"].notnull()].iterrows():
        annotate_period(ax, row["label"], row["date1"], row["date2"])
    
    #LOGGER.info(f"Writing to {output_file}")
    #file_ext = output_file.split(".")[-1]
    #plt.savefig(output_file, format=file_ext)
    plt.show()


if __name__ == "__main__":
    typer.run(main)
