import os
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_webpages(output_folder):
    # Fetch content from a URL
    def fetch_text_from_webpage(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None

    # Clean and preprocess text
    def clean_and_preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        return " ".join(tokens)  # Join tokens into a single string

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the list of URLs from "links.txt"
    with open("links.txt", "r") as links_file:
        urls = links_file.read().splitlines()

    for url in urls:
        # Main preprocessing function for each URL
        webpage_text = fetch_text_from_webpage(url)
        if webpage_text:
            soup = BeautifulSoup(webpage_text, 'html.parser')
            plain_text = soup.get_text()
            preprocessed_text = clean_and_preprocess_text(plain_text)

            # Find the last document number
            file_list = os.listdir(output_folder)
            file_list = [file for file in file_list if file.startswith("preprocessed_webpage")]
            file_numbers = [int(re.search(r'\d+', file).group()) for file in file_list]
            if file_numbers:
                new_file_number = max(file_numbers) + 1
            else:
                new_file_number = 1

            # Define the new filename
            new_filename = f"preprocessed_webpage{new_file_number}.txt"
            new_filepath = os.path.join(output_folder, new_filename)

            # Save preprocessed text to the new file
            with open(new_filepath, "w") as file:
                file.write(preprocessed_text)
            print(f"Preprocessed text from '{url}' saved to '{new_filename}'")
        else:
            print(f"Failed to fetch or preprocess the content from '{url}'.")


output_folder = 'preprocessed_data' 
preprocess_webpages(output_folder)


"""
https://indiankanoon.org/doc/78724635/
https://indiankanoon.org/doc/62111311/
https://indiankanoon.org/doc/15749014/
https://indiankanoon.org/doc/131833088/
https://indiankanoon.org/doc/116958455/
https://indiankanoon.org/doc/29610717/
https://indiankanoon.org/doc/164902416/
https://indiankanoon.org/doc/115830876/
https://indiankanoon.org/doc/73866393/
https://indiankanoon.org/doc/810217/
https://indiankanoon.org/doc/84956611/
https://indiankanoon.org/doc/53997832/
https://indiankanoon.org/doc/1036814/
https://indiankanoon.org/doc/116159756/
https://indiankanoon.org/doc/142233676/
https://indiankanoon.org/doc/79551006/
https://indiankanoon.org/doc/100493410/
https://indiankanoon.org/doc/151168168/
https://indiankanoon.org/doc/72413080/
https://indiankanoon.org/doc/175854279/
https://indiankanoon.org/doc/110121958/
"""





