import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_selection import SelectKBest, f_regression

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
bert_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define function to tokenize and extract BERT embeddings for each text
def get_bert_embeddings(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Define file path to preprocessed data folder
preprocessed_folder = 'preprocessed_data'

# Get list of preprocessed webpage files
preprocessed_files = os.listdir(preprocessed_folder)

# Select a file to use
file_name = 'preprocessed_webpage1.txt'

# Load data from selected file
file_path = os.path.join(preprocessed_folder, file_name)
with open(file_path, 'r') as f:
    text_data = f.read().splitlines()

# Create a DataFrame containing the text data
data = pd.DataFrame({'text': text_data})

# Tokenize and extract BERT embeddings
data['bert_embeddings'] = data['text'].apply(get_bert_embeddings)

# Feature selection
X = np.vstack(data['bert_embeddings'])
selector = SelectKBest(score_func=f_regression, k=10)  # You can choose the number of features (k) as needed
X_selected = selector.fit_transform(X, None)  # Pass None as the second argument for unsupervised feature selection

# Save X_selected to a file
output_file = 'selected_features.npy'
np.save(output_file, X_selected)