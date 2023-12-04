import urllib.request

# Download files
urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research/bert/master/modeling.py", "modeling.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research/bert/master/optimization.py", "optimization.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py", "run_classifier.py")
urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research/bert/master/tokenization.py", "tokenization.py")


import numpy as np # linear algebra
import re, os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf

import datetime
from datetime import datetime

# BERT
from tensorflow.keras.optimizers import Adam
import run_classifier
import tokenization
import tensorflow_hub as hub



directory = []
file = []
title = []
text = []
label = []

datapath = '../data/'

for dirname, _, filenames in os.walk(datapath):
    # Skip the README.TXT file if present
    try:
        filenames.remove('README.TXT')
    except:
        pass
    
    for filename in filenames:
        directory.append(dirname)
        file.append(filename)
        label.append(dirname.split('/')[-1])
        fullpathfile = os.path.join(dirname, filename)
        
        with open(fullpathfile, 'r', encoding="utf8", errors='ignore') as infile:
            intext = ''
            firstline = True
            
            for line in infile:
                if firstline:
                    title.append(line.replace('\n', '').strip())
                    firstline = False
                else:
                    intext = intext + ' ' + line.replace('\n', '')
            
            text.append(intext)

DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

fulldf = pd.DataFrame(list(zip(directory, file, title, text, label)), 
               columns =['directory', 'file', 'title', 'text', 'label'])

df = fulldf.filter(['text','label'], axis=1)
df.head()