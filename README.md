# messenger

Facebook allows us to download our own messenger data, giving a set of JSON files for each chat history. The goal here is to pull out the meaningful information from a given chat and produce visualisations of the data. Some plots make for nice wallpaper images (and don't attempt to convey accurate information) and some show interesting trends in the data.


## Getting Started

### Getting the data
The first step is to download a copy of your data from Facebook at the following page:

'''
Settings > Your Facebook information > Download your information
'''

Select "Messages" and ensure the format is "JSON" before clicking "Create File".
Once this is downloaded and extracted, the JSON files can be found in the <chat name> folder within the "inbox" folder.


### Prerequisites
The package_list.txt file contains all packages installed into the working environment and which versions are used. Key libraries are: matplotlib, Pandas, NumPy, NLTK.

### The programs
* Program load_and_preprocess.py produces a Pandas DataFrame where each entry is a message sent in the chat (along with relevant information).
* Program plot_artsy_anon.py produces a plot of total word count against date for use as a wallpaper.
* Program plot_polar.py produces plots of word (or message) count against time of day, day of the week, or month of the year as a polar plot.


## Acknowledgements 
The function to produce NLTK-ised versions of text is largely based on a blog post by Hunter Heidenreich (with edits/additions by myself).