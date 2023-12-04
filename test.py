import os
import requests
from bs4 import BeautifulSoup

def fetch_text_from_webpage(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def preprocess_webpages(output_folder):
    data_folder = os.path.join(output_folder, '..', 'data', 'not_cybercrime')

    # Create the output folder structure if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Read the list of URLs from "links.txt"
    with open("links.txt", "r") as links_file:
        urls = links_file.read().splitlines()

    for url in urls:
        # Main preprocessing function for each URL
        webpage_text = fetch_text_from_webpage(url)
        if webpage_text:
            soup = BeautifulSoup(webpage_text, 'html.parser')
            plain_text = soup.get_text()

            # Find the last document number
            file_list = os.listdir(data_folder)
            file_list = [file for file in file_list if file.startswith("raw_webpage")]
            file_numbers = [int(file.split('.')[0].split('webpage')[-1]) for file in file_list]
            if file_numbers:
                new_file_number = max(file_numbers) + 1
            else:
                new_file_number = 1

            # Define the new filename
            new_filename = f"raw_webpage{new_file_number}.txt"
            new_filepath = os.path.join(data_folder, new_filename)

            # Save raw text to the new file
            with open(new_filepath, "w", encoding='utf-8') as file:
                file.write(plain_text)
            print(f"Raw text from '{url}' saved to '{new_filename}'")
        else:
            print(f"Failed to fetch or preprocess the content from '{url}'.")

output_folder = 'preprocessed_data'  # Change this if needed
preprocess_webpages(output_folder)
