from datasets import load_from_disk
import pickle
import os
from tqdm import tqdm
import pm4py
import argparse
import re

import json
import time
from dotenv import load_dotenv

from configGenAI import init

# Load environment variables from .env file
load_dotenv()

openai = init()

dataset='sap_sam_2022/filtered'

def _get_constraint_description(constraint_type, activity_a="A", activity_b="B"):
    """
    Get a description for the given DECLARE constraint type.

    Parameters:
    - constraint_type (str): The type of DECLARE constraint (e.g., 'Precedence', 'Response').
    - activity_a (str, optional): The first activity in the constraint. Default is A.
    - activity_b (str, optional): The second activity in the constraint. Default is B.

    Returns:
    - str: A human-readable description of the constraint.
    """
    descriptions = {
        "Alternate Precedence": f"Alternate Precedence({activity_a}, {activity_b}) means that if activity {activity_b} happens, activity {activity_a} must happen before, and no other occurrences of {activity_b} may intervene.",
        "Alternate Response": f"Alternate Response({activity_a}, {activity_b}) means that each occurrence of activity {activity_a} must be followed by activity {activity_b}, and no other occurrences of {activity_a} may intervene.",
        "Alternate Succession": f"Alternate Succession({activity_a}, {activity_b}) means that each occurrence of {activity_a} must be followed by {activity_b}, and each occurrence of {activity_b} must be preceded by {activity_a}, with no intervening occurrences.",
        "Choice": f"Choice({activity_a}, {activity_b}) means that either activity {activity_a} or {activity_b} must occur, but not necessarily both.",
        "Co-Existence": f"Co-Existence({activity_a}, {activity_b}) means that if {activity_a} occurs, then {activity_b} must also occur, and vice versa.",
        "End": f"End({activity_a}) means that activity {activity_a} must be the last activity in the process.",
        "Exclusive Choice": f"Exclusive Choice({activity_a}, {activity_b}) means that either {activity_a} or {activity_b} can occur, but not both.",
        "Init": f"Init({activity_a}) means that activity {activity_a} must be the first activity in the process.",
        "Precedence": f"Precedence({activity_a}, {activity_b}) means that if activity {activity_b} happens, activity {activity_a} must happen before.",
        "Response": f"Response({activity_a}, {activity_b}) means that if activity {activity_a} happens, activity {activity_b} must eventually happen after.",
        "Succession": f"Succession({activity_a}, {activity_b}) means that if activity {activity_a} happens, activity {activity_b} must eventually happen after, and vice versa.",
    }

    # Return the corresponding description or a default message
    return descriptions.get(
        constraint_type,
        f"No description available for constraint type: {constraint_type}.",
    )


def predict_with_openai(llm, constraint_type, eventlog, all_constraints_at_once, textdesc = None):
    """
    Uses GPT-4o via OpenAI API (through GenAIHubProxyClient) to generate predictions of DECLARE constraints for Eventlog.
    """
    if all_constraints_at_once:
        constraints_of_interest = [
            'Alternate Precedence',
            'Alternate Response',
            'Alternate Succession',
            'Choice',
            'Co-Existence',
            'End',
            'Exclusive Choice',
            'Init',
            'Precedence',
            'Response',
            'Succession'
        ]
        if textdesc:
            # Prompt combining eventlog and textdesc to generate list of predicted constraints
            messages = [
                {"role": "system", "content": "You are an expert in process mining and DECLARE constraint discovery."},
                {"role": "user", "content": (
                    "Given the following list of unique eventlog activities in a business process its associated natural language description of the process, "
                    "predict a list of likely DECLARE constraints that best represent the process. Focus only on the following "
                    "types of constraints:\n"
                    f"{', '.join(constraints_of_interest)}.\n\n"
                    "Provide the constraints in the standard format (e.g., Succession[A, B], Choice[A, B]). "
                    "Do not include any introductions, explanations, or additional text. Just output the constraints as a plain list:\n\n"
                    f"Event Log:\n{eventlog}\n\n"
                    f"Text Description:\n{textdesc}\n"
                )}
            ]
        else:
            # Prompt only using eventlog to generate list of predicted constraints
            messages = [
                {"role": "system", "content": "You are an expert in process mining and DECLARE constraint discovery."},
                {"role": "user", "content": (
                    "Given the following list of unique eventlog activities in a business process, predict a list of likely DECLARE constraints that best represent the process. "
                    "Focus only on the following types of constraints:\n"
                    f"{', '.join(constraints_of_interest)}.\n\n"
                    "Provide the constraints in the standard format (e.g., Succession[A, B], Choice[A, B]). "
                    "Do not include any introductions, explanations, or additional text. Just output the constraints as a plain list:\n\n"
                    f"Event Log:\n{eventlog}\n"
                )}
            ]

    else:
        constraint_desc = _get_constraint_description(constraint_type)
        if textdesc:
            # Prompt combining eventlog and textdesc to generate list of predicted constraints
            messages = [
                {"role": "system", "content": "You are an expert in process mining and DECLARE constraint discovery."},
                {"role": "user", "content": (
                    "Given the following list of unique eventlog activities in a business process and its associated natural language description of the process, "
                    "predict a list of likely DECLARE constraints that best represent the process. Focus only on the following "
                    "type of constraint:\n"
                    f"{constraint_desc}\n\n"
                    "Provide the constraints in the standard format (e.g., Succession[A, B], Choice[A, B]). "
                    "Do not include any introductions, explanations, or additional text. Just output the constraints as a plain list:\n\n"
                    f"Event Log:\n{eventlog}\n\n"
                    f"Text Description:\n{textdesc}\n"
                )}
            ]
        else:
            # Prompt only using eventlog to generate list of predicted constraints
            messages = [
                {"role": "system", "content": "You are an expert in process mining and DECLARE constraint discovery."},
                {"role": "user", "content": (
                    "Given the following list of unique eventlog activities in a business process, predict a list of likely DECLARE constraints that best represent the process. "
                    "Focus only on the following type of constraint:\n"
                    f"{constraint_desc}.\n\n"
                    f"Provide the constraints in the standard format ({constraint_type}[A, B]). "
                    "Do not include any introductions, explanations, or additional text. Just output the constraints as a plain list:\n\n"
                    f"Event Log:\n{eventlog}\n"
                )}
            ]
            
    try:
        response = openai.chat.completions.create(
            model_name=llm,
            messages=messages,
            temperature=0,
            max_tokens=2000,  # Adjust token limit as needed
            frequency_penalty=0,
            top_p=1,
            seed=1234
        )
    
        # Access the message content properly
        out = json.loads(response.json())
        result = out["choices"][0]["message"]["content"]
        predictions = [re.sub(r'^[^a-zA-Z]*', 
                              '', line.strip()) for line in result.split("\n") 
                              if "[" in line and "]" in line]

        
        return predictions
    except Exception as e:
        return f"Error generating description: {e}"

def process_files(llm, case_names, eventlogs_folder, output_folder, all_constraints_at_once,textdescs_folder = None):
    """
    Processes all files in the input folders and saves constraints as .pkl files in the output folder.
    """
    print(output_folder)
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    counter = 0 
    nr_logs = len(case_names)
    for file_name in case_names:
        counter += 1
        print(file_name)
        case_name = file_name.split('.')[0]
        eventlog_path = os.path.join(eventlogs_folder, case_name + ".xes")
        print(f"Processing file: {case_name}. Nr: {counter}/{nr_logs}")
        try:
            eventlog = pm4py.read_xes(eventlog_path)
            eventlog = pm4py.convert_to_dataframe(eventlog)
            eventlog = eventlog["concept:name"].unique()
            #print(eventlog)

            textdesc = None
            if(textdescs_folder):
                textdesc_path = os.path.join(textdescs_folder, case_name + ".txt")

                with open(textdesc_path, 'rb') as file:
                    textdesc = file.readlines() 
            
            relevant_data_dir = f'data/{dataset}/constraints_to_log_labels/'
            # Attempt to open the constraints file
            path_to_constraints = os.path.join(relevant_data_dir, f'{case_name}.CONSTRAINTS.pkl')
            with open(path_to_constraints, 'rb') as f:
                constraints = pickle.load(f)

            # Process constraints
            result_list = []
            all_constraint_types_in_model = list(set([i.split('[')[0] for i in constraints]))
            for constraint_type in all_constraint_types_in_model:
                predictions = predict_with_openai(llm, constraint_type, eventlog, all_constraints_at_once, textdesc)

                # Append the results for the current constraint type
                result_list.extend(predictions)

            # Save predictions to file
            file_name_path = f'{output_folder}{case_name}.pkl'
            with open(file_name_path, 'wb') as f:
                pickle.dump(result_list, f)
            
            # To print only first input/output uncomment bellow
            """
            print(f"Description saved to {output_folder}")
            print(eventlog)
            print("---------Predicted constraints------------")
            print(result_list)
            exit()
            """
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")


def run_model(llm, args, output_dir, model_case_names):
    """
    Runs the model processing based on specified arguments.
    """
    prediction_output_dir = f"{output_dir}" + llm
    eventlogs_folder = f"data/{dataset}/logs"
    textdescs_folder = None
    
    # Adjust folder paths based on whether text descriptions are used
    if args.use_textdescs:
        textdescs_folder = f"data/{dataset}/textdescs"
        prediction_output_dir += "_combined/"
        if not os.path.exists(textdescs_folder):
            print("Error: The text descriptions folder does not exist.")
            print("You need to run declare_to_textdesc.py file before this to generate the textdescs.")
            exit(1)
    else:
        prediction_output_dir += "_eventlog/"

    if os.path.exists(eventlogs_folder):
        # Start timing
        start_time = time.time()

        # Process files
        process_files(llm, model_case_names, eventlogs_folder, prediction_output_dir, args.at_once, textdescs_folder)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTranslation completed in {elapsed_time:.2f} seconds.")
    else:
        print("The specified input folder does not exist.")


def main():

    #load train cases
    dataset_for_training_dir = f'data/{dataset}/forTraining'#'data/bpmai/forTraining_flan_t5'
    training_dataset_filename = 'training'
    path_to_training_dataset = os.path.join(dataset_for_training_dir, training_dataset_filename)
    data = load_from_disk(path_to_training_dataset)
    validation_data = data['test'].to_pandas()
    model_case_names = validation_data.id.unique() 

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process event logs with optional settings.")
    parser.add_argument("--use_textdescs", action="store_true", help="Use text descriptions folder during processing.")
    parser.add_argument("--llama", action="store_true", help="Use Llama instead of GPT-4o.")
    parser.add_argument("--everything", action="store_true", help="Run all combinations of models and with/without textdesc")
    parser.add_argument("--at_once", action="store_true", help="Prompt LLM with all constrainttypes as ones")
    parser.add_argument("-n", type=int, help="Specify the number for nr of times processing (e.g., -n4 for 4).")

    args = parser.parse_args()
    # Accessing the value of -n
    if args.n is not None:
        print(f"Nr of times to run the model: {args.n}")
        times_to_run = args.n
    else:
        print("No specific number provided; proceeding with default one time.")
        times_to_run = 1
    for i in range(times_to_run):
        prediction_output_dir = f'data/evaluation/{dataset}/test{i+1}/'
        if args.at_once:
            prediction_output_dir += "at_once/"
        prediction_output_dir += "vanilla_"
        if args.everything:
            # Run all combinations
            for llm in ["gpt-4o", "meta--llama3.1-70b-instruct"]:
                for use_textdescs in [False, True]:
                    args.use_textdescs = use_textdescs
                    run_model(llm, args, prediction_output_dir, model_case_names)
        else:
            # Choose model based on argument
            if args.llama:
                llm = "meta--llama3.1-70b-instruct"
            else:
                llm = "gpt-4o"
            
            # Run based on text description argument
            run_model(llm, args, prediction_output_dir, model_case_names)


if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    main()
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nEverything completed in {elapsed_time:.2f} seconds.")