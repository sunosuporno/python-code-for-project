import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_webpage(url):
    # Fetch content from the URL
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

    # Main preprocessing function
    webpage_text = fetch_text_from_webpage(url)
    if webpage_text:
        soup = BeautifulSoup(webpage_text, 'html.parser')
        plain_text = soup.get_text()
        preprocessed_text = clean_and_preprocess_text(plain_text)
        return preprocessed_text
    else:
        return None

# Example usage
webpage_url = 'https://indiankanoon.org/doc/148743692/'  # Replace with your actual URL
preprocessed_text = preprocess_webpage(webpage_url)
if preprocessed_text:
    # Save preprocessed text to a text file
    with open("preprocessed_webpage2.txt", "w") as file:
        file.write(preprocessed_text)
    print("Preprocessed text saved to 'preprocessed_webpage2.txt'")
else:
    print("Failed to fetch or preprocess the content from the webpage.")
