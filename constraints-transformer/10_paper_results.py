import pandas as pd
import pickle
import os
from glob import glob

model_checkpoint = "checkpoint-12276"
model_name='google/flan-t5-small'
dataset='sap_sam_2022/filtered'
# Define the path to the output directory where .pkl files for test scores are stored
prediction_output_dir = f'data/evaluation/{dataset}/test/{model_name}_{model_checkpoint}/'

# Initialize an empty DataFrame to accumulate results from all files
all_results = []

# Loop through all .pkl files in the directory
for file_path in glob(os.path.join(prediction_output_dir, '*.pkl')):
    # Load each .pkl file
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    
    print(results)
    # Convert each result into a DataFrame with the expected structure
    for constraint_type, predictions in results:
        for pred in predictions:
            print(pred)
            precision, recall, f1 = pred  # Access by position
            all_results.append({
                'constraint_type': constraint_type,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

# Convert the accumulated list into a DataFrame
df = pd.DataFrame(all_results)

# Group by 'constraint_type' and calculate mean precision, recall, and f1, rounded to 2 decimals
evaluation_scores = df.groupby('constraint_type').mean().reset_index().round(2)

# Display the results
print(evaluation_scores)