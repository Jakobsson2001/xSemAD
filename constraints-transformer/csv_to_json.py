import os
import pandas as pd
import json

from config import config_preprocessing_param

# Step 1: Load the CSV file
csv_file_path = 'evaluation/semantic_sap_sam_filtered.csv'  # Replace with your actual CSV file path
output_folder = config_preprocessing_param['json_dir']  # Folder to store the JSON files

# Step 2: Create a folder to store JSON files if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 3: Read the CSV file with encoding fallback (utf-8 and then ISO-8859-1)
try:
    df = pd.read_csv(csv_file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Step 4: Iterate through the rows and save each JSON object to a file
for index, row in df.iterrows():
    json_data = row['Model JSON']  # Access the JSON data as a string
    file_name = f"{row['Model ID']}_{row['Revision ID']}.json"  # Filename includes both Model ID and Revision ID
    # Parse the JSON string into a dictionary
    try:
        json_object = json.loads(json_data)
        
        # Step 5: Write the JSON object to a file
        with open(os.path.join(output_folder, file_name), 'w') as json_file:
            json.dump(json_object, json_file, indent=4)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for Model ID {row['Model ID']} and Revision ID {row['Revision ID']}")


print(json_data) 
print("JSON files have been saved.")
