import spacy
from transformers import pipeline, BertTokenizer, BertForTokenClassification
import torch
import os

def extract_info_bert(doc):
    convicts = set()
    court = None
    judgment_given = False

    # Use a BERT model for sentiment analysis
    sentiment_analyzer = pipeline('sentiment-analysis')

    # Use a BERT model for named entity recognition
    ner_model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

    # Iterate through sentences in the document
    for sent in doc.sents:
        # Check sentiment of the sentence
        sentiment_result = sentiment_analyzer(sent.text)
        sentiment_score = sentiment_result[0]['score']

        # If sentiment is negative, consider it for information extraction
        if sentiment_score < 0.5:  # You can adjust this threshold based on your needs
            # Tokenize the sentence for named entity recognition
            tokens = tokenizer(sent.text, return_tensors="pt")
            outputs = ner_model(**tokens)

            # Extract named entities
            predictions = torch.argmax(outputs.logits, dim=2)
            entities = tokenizer.batch_decode(predictions[0])

            for token, entity in zip(tokens['input_ids'][0], entities[1:-1]):  # Exclude [CLS] and [SEP]
                token_text = tokenizer.decode(token.item()).strip()

                # Check for the convict's name (PERSON entity)
                if 'I-PER' in entity:
                    convicts.add(token_text)

                # Check for the court name (ORG entity)
                elif 'I-ORG' in entity:
                    court = token_text

                # Check for keywords indicating judgment
                elif token_text.lower() in ['judgment', 'verdict', 'sentence']:
                    judgment_given = True

    return {'convicts': list(convicts), 'court': court, 'judgment_given': judgment_given}

def process_text_file(file_path):
    # Read the content of the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()

    # Process the text using spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text_data)

    # Extract information using the defined function
    result = extract_info_bert(doc)

    # Print the extracted information
    print(result)

# Specify the path to the preprocessed text file
file_path = os.path.join('data', 'cybercrime', 'raw_webpage3.txt')

# Process the text file
process_text_file(file_path)
