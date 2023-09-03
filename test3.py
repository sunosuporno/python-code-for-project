import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import csv  # Import CSV module

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Clean and preprocess text
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def count_words_and_numbers(folder_path):
    word_counter = Counter()
    number_counter = Counter()

    # Iterate through preprocessed files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = preprocess_text(text)

                for token in tokens:
                    # Check if the token is a number (using regular expression)
                    if re.match(r'^-?\d+(\.\d+)?$', token):
                        number_counter[token] += 1
                    else:
                        word_counter[token] += 1

    # Filter words and numbers that occur more than 10 times
    frequent_words = [(word, count) for word, count in word_counter.items() if count > 10]
    frequent_numbers = [(num, count) for num, count in number_counter.items() if count > 10]

    return frequent_words, frequent_numbers

# Example usage
folder_path = 'preprocessed_data'  # Folder where the preprocessed files are stored
frequent_words, frequent_numbers = count_words_and_numbers(folder_path)

# Write results to a CSV file
with open('frequent_words_and_numbers.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    if frequent_words:
        csv_writer.writerow(["Frequent Words", "Frequency"])
        csv_writer.writerows(frequent_words)
    
    if frequent_numbers:
        csv_writer.writerow(["Frequent Numbers", "Frequency"])
        csv_writer.writerows(frequent_numbers)

# Print results with frequency
if frequent_words:
    print("Frequent Words with Frequency:")
    for word, freq in frequent_words:
        print(f"{word}: {freq}")

if frequent_numbers:
    print("\nFrequent Numbers with Frequency:")
    for num, freq in frequent_numbers:
        print(f"{num}: {freq}")

print("\nResults written to 'frequent_words_and_numbers.csv'.")
