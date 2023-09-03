import nltk
from nltk import FreqDist
nltk.download('punkt')  # Download the necessary resources if not already downloaded

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
    input_filename = "preprocessed_webpage4.txt"  # Replace with the path to your tokenized input text file
    
    # Read the tokenized input from the text file
    with open(input_filename, "r") as file:
        input_text = file.read()
    
    # List of keywords
    keywords = [
        "cyberbully", "online harassment", "porn", "sextort",
        "stalk", "grooming", "digital abuse", "dox", "doxx", "Intimate image abuse", "identity theft",
        "phish", "online fraud", "catfish", "dating",
        "online trafficking", "online exploitation", "cyber", "cybercrime"
        "hate speech", "misogyny", "cyber stalk", "troll",
        "voyeurism", "digital manipulation", "impersonation",
        "online vigilante justice", "354 A", "354 a", "354 D", "345 d", "354 C", "354 c"    
        ]
    
    # Preprocess the input text
    tokens = preprocess_text(input_text)
    
    # Extract keyword features
    keyword_counts = extract_features(tokens, keywords)
    
    # Print the keyword counts
    for keyword, count in keyword_counts.items():
        if count > 0:
            print(f"{keyword}: {count}")
