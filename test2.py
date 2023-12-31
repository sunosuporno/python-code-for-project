import os
import nltk
from nltk import FreqDist
nltk.download('punkt')

def preprocess_text(text):
    # Tokenize the input text
    tokens = nltk.word_tokenize(text)
    return tokens

def extract_features(tokens, keywords):
    # Create a frequency distribution of tokens
    fdist = FreqDist(tokens)

    # Initialize a dictionary to store the keyword counts
    keyword_counts = {keyword: 0 for keyword in keywords}

    # Count the occurrences of each keyword in the tokens
    for keyword in keywords:
        keyword_counts[keyword] = fdist[keyword]

    return keyword_counts

if __name__ == "__main__":
    folder_path = "preprocessed_data"
    input_filename = "preprocessed_webpage20.txt"

    # Construct the full path to the input file
    input_filepath = os.path.join(folder_path, input_filename)

    # Read the tokenized input from the text file
    with open(input_filepath, "r") as file:
        input_text = file.read()

    # List of keywords (replace with your own keywords)
    keywords = [
        "cyberbully", "harassment", "porn", "sextort",
        "stalk", "grooming", "digital abuse", "dox", "doxx", "abus", "identity theft",
        "phish", "fraud", "catfish",  "radical", "insurrect", "terror", "violent overthrow",
        "trafficking", "extort", "cyber", "cybercrime", "499", "509", "507", "303"
        "hate", "speech", "misogyn", "cyber stalk", "troll", "secess", "treason", "sedit"
        "voyeur", "manipulat", "imperson" ,"Cyberattack","Darkweb","Databreach","Phishing",
        "Malware", "theft", "DOS", "Ransomware","Internetfraud","theft", "Clickjacking","Spoofing","Deepfakes", 
        "Hacking", "Carding", "Information security", "67", "66", "botnet", "354", "Anti-national"
    ]

    # Preprocess the input text
    tokens = preprocess_text(input_text)

    # Extract keyword features
    keyword_counts = extract_features(tokens, keywords)

    # Print the keyword counts
    for keyword, count in keyword_counts.items():
        if count > 0:
            print(f"{keyword}: {count}")
