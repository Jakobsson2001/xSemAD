import random
import itertools
import os
import pickle
import argparse

from datasets import load_from_disk

def generate_constraints(constraint_types, labels):
    constraints = []

    #print(labels)
    for constraint_type in constraint_types:
        #print(constraint_type)
        # Generate all possible combinations for this constraint type
        if constraint_type == "Init" or constraint_type == "End":
            combinations = list(itertools.permutations(labels, 1))
        else:
            combinations = list(itertools.permutations(labels, 2))

        #print(constraint_type)
        #print(combinations)
        
        # Calculate max number of constraints to generate
        max_constraints = max(1, int(len(combinations) * 0.3))
        #max_constraints = len(combinations)
        #print(max_constraints)
        num_constraints = random.randint(0, max_constraints)
        #print("num const", num_constraints)

        # Randomly select combinations
        #print(combinations)
        #print(num_constraints)
        if len(combinations) > 0:
            selected_combinations = random.sample(combinations, num_constraints)

            # Generate the constraint strings
            for combination in selected_combinations:
                constraint = f"{constraint_type}[{', '.join(combination)}]"
                constraints.append(constraint)

    return constraints


dataset='sap_sam_2022/filtered'

#load train cases
dataset_for_training_dir = f'data/{dataset}/forTraining'#'data/bpmai/forTraining_flan_t5'
training_dataset_filename = 'training'
path_to_training_dataset = os.path.join(dataset_for_training_dir, training_dataset_filename)
data = load_from_disk(path_to_training_dataset)
validation_data = data['test'].to_pandas()
model_case_names = validation_data.id.unique() 

relevant_data_dir = f'data/{dataset}/constraints_to_log_labels/'
 
# Argument parser setup
parser = argparse.ArgumentParser(description="Generate random constraints for process models.")
parser.add_argument("-n", type=int, help="Specify the number for number of times to generate random constraints for each model (e.g., -n4 for 4).")

args = parser.parse_args()
# Accessing the value of -n
if args.n is not None:
    print(f"Nr of times to run the model: {args.n}")
    times_to_run = args.n
else:
    print("No specific number provided; proceeding with default one time.")
    times_to_run = 1

for i in range(times_to_run):
    output_folder = f'data/evaluation/{dataset}/test{i+1}/random/'
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    for case_name in model_case_names:
        # Attempt to open the constraints file
        path_to_constraints = os.path.join(relevant_data_dir, f'{case_name}.CONSTRAINTS.pkl')
        with open(path_to_constraints, 'rb') as f:
            constraints = pickle.load(f)

        path_to_labels = os.path.join(relevant_data_dir, f'{case_name}.LABELS.pkl')
        with open(path_to_labels, 'rb') as f:
            labels = pickle.load(f)

        # Process constraints
        all_constraint_types_in_model = list(set([i.split('[')[0] for i in constraints]))
        # Generate and print constraints
        constraints = generate_constraints(all_constraint_types_in_model, labels)
            
        # Save predictions to file
        file_name_path = f'{output_folder}{case_name}.pkl'
        with open(file_name_path, 'wb') as f:
            pickle.dump(constraints, f)

        """
        print(f"-----------{case_name}-------------")
        print(constraints)
        print("------------------------------------")
        #print("\n".join(constraints))
        """

