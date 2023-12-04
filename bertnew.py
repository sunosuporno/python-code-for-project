import urllib.request
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

# Download BERT model from TensorFlow Hub
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
bert_layer = hub.KerasLayer(bert_model_url, trainable=True)

# Additional imports for preprocessing
from transformers import BertTokenizer

# Load your data
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

# Create DataFrame
DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

fulldf = pd.DataFrame(list(zip(directory, file, title, text, label)), 
               columns =['directory', 'file', 'title', 'text', 'label'])

df = fulldf.filter([DATA_COLUMN, 'label'], axis=1)
df.head()

# Preprocess the text data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

def bert_preprocess(texts, tokenizer, max_len=128):
    print('preprocessing...')
    print(texts.tolist())
    tokens = tokenizer(texts.tolist(), max_length=max_len, truncation = True, padding = True, return_tensors='tf')
    return {key: tokens[key] for key in tokens}

X = bert_preprocess(df[DATA_COLUMN], tokenizer, max_len=max_len)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, df[LABEL_COLUMN].values, test_size=0.2, random_state=42)

# Build the model using BERT
def build_model(bert_layer, max_len=128):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    bert_inputs = dict(input_word_ids=input_word_ids)
    x = bert_layer(bert_inputs)["pooled_output"]
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_word_ids, outputs=x)
    model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Build and train the model
model = build_model(bert_layer, max_len=max_len)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)

# Evaluate the model
model.evaluate(X_test, y_test)
