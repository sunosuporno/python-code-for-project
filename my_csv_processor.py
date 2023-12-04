import os
import pandas as pd

def process_text_files(input_folder, output_csv):
    # Create or check 'data' folder
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Create or check 'cybercrime' and 'not_cybercrime' folders
    cybercrime_folder = os.path.join(data_folder, 'cybercrime')
    not_cybercrime_folder = os.path.join(data_folder, 'not_cybercrime')
    for folder in [cybercrime_folder, not_cybercrime_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Process text files and create DataFrame
    data = {'text': [], 'type': []}

    for folder, label in [(cybercrime_folder, 'cybercrime'), (not_cybercrime_folder, 'not_cybercrime')]:
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as text_file:
                content = text_file.read()
                data['text'].append(content)
                data['type'].append(label)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(__file__), 'data')
    output_csv = os.path.join(os.path.dirname(__file__), 'output.csv')
    process_text_files(input_folder, output_csv)
    print(f"CSV file '{output_csv}' generated successfully.")
