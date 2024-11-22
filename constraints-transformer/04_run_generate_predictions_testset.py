from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import load_from_disk
import pickle
import os
import numpy as np 
from tqdm import tqdm
import pandas as pd
from evaluation.utils import generate_prediction_list, sort_constraints, filter_prediction_list_for_eval,sort_constraints_for_eval
from labelparser.label_utils import constraint_splitter
import torch

#below would be for the fetched from zendo?
#model_checkpoint =  "checkpoint-127200/checkpoint-42400"#"checkpoint-118800"
model_checkpoint = "checkpoint-25848"
model_name='google/flan-t5-small'
dataset='sap_sam_2022/filtered'
max_new_tokens=100
prediction_output_dir = f'data/evaluation/{dataset}/test/{model_name}_{model_checkpoint}/'

#load train cases
dataset_for_training_dir = f'data/{dataset}/forTraining'#'data/bpmai/forTraining_flan_t5'
training_dataset_filename = 'training'
path_to_training_dataset = os.path.join(dataset_for_training_dir,training_dataset_filename)
data = load_from_disk(path_to_training_dataset)
validation_data = data['test'].to_pandas()
model_case_names = validation_data.id.unique() 

#get all possible constraint types
constraint_type='DECLARE'
constraints_dir = f'data/{dataset}/constraints'

#load model
model_dir = f"data/model/{dataset}/{model_name}/{model_checkpoint}"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
#to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# 
relevant_data_dir = f'data/{dataset}/constraints_to_log_labels/'

if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)

filesNotFound = 0
for model_case_name in tqdm(model_case_names, desc='make predictions'):
    result_list = []
    try:
        # Attempt to open the labels file
        path_to_labels = os.path.join(relevant_data_dir, f'{model_case_name}.LABELS.pkl')
        with open(path_to_labels, 'rb') as f:
            labels = list(pickle.load(f))

        # Attempt to open the constraints file
        path_to_constraints = os.path.join(relevant_data_dir, f'{model_case_name}.CONSTRAINTS.pkl')
        with open(path_to_constraints, 'rb') as f:
            constraints = pickle.load(f)
        
        # Process constraints
        all_constraint_types_in_model = list(set([i.split('[')[0] for i in constraints]))
        for constraint_type in all_constraint_types_in_model:
            # Prepare context for prediction generation
            context = constraint_type +": <event>" + "<event>".join(labels)
            
            # Filter and sort true constraints for the current type
            true_constraints = [c for c in constraints if c.startswith(f"{constraint_type}[")]
            true_constraints = sort_constraints(true_constraints, remove_duplicates=True)
            
            # Generate predictions based on the context
            predictions = generate_prediction_list(
                context, tokenizer, model, 30, 
                max_new_tokens=max_new_tokens, device=device
            )
            
            # Filter and sort predictions for evaluation
            predictions = filter_prediction_list_for_eval(model_labels=labels, prediction_c_list=predictions)
            predictions = sort_constraints_for_eval(predictions, remove_duplicates=True)
            
            # Append the results for the current constraint type
            result_list.append((constraint_type, predictions))

        # Save predictions to file
        file_name_path = f'{prediction_output_dir}{model_case_name}.pkl'
        with open(file_name_path, 'wb') as f:
            pickle.dump(result_list, f)

    except (FileNotFoundError, IOError) as e:
        # Handle the error and continue to the next iteration
        filesNotFound += 1
        #print(f"Could not process {model_case_name}: {e}")
        continue

print('DONE! And there where ', filesNotFound, ' files not found')
