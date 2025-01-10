from datasets import load_from_disk
import pickle
import os
from tqdm import tqdm
import pm4py
import argparse

import json
import time
from dotenv import load_dotenv

from configGenAI import init

# Load environment variables from .env file
load_dotenv()

openai = init()

def predict_with_openai(llm, eventlog, textdesc = None):
    """
    Uses GPT-4o via OpenAI API (through GenAIHubProxyClient) to generate predictions of DECLARE constraints for Eventlog.
    """
    try:
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
        predictions = [line.strip() for line in result.split("\n") if "[" in line and "]" in line]

        return predictions
    except Exception as e:
        return f"Error generating description: {e}"

def process_files(llm, case_names, eventlogs_folder, output_folder, textdescs_folder = None):
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
            
            predictions = predict_with_openai(llm, eventlog, textdesc)

            # Save predictions to file
            file_name_path = f'{output_folder}{case_name}.pkl'
            with open(file_name_path, 'wb') as f:
                pickle.dump(predictions, f)
            
            # To print only first input/output uncomment bellow
            print(f"Description saved to {output_folder}")
            print(eventlog)
            print("---------Predicted constraints------------")
            print(predictions)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")


def run_model(llm, use_textdescs, dataset, model_case_names):
    """
    Runs the model processing based on specified arguments.
    """
    prediction_output_dir = f"output/{dataset}/" + llm
    eventlogs_folder = f"data/{dataset}/logs"
    textdescs_folder = None
    
    # Adjust folder paths based on whether text descriptions are used
    if use_textdescs:
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
        process_files(llm, model_case_names, eventlogs_folder, prediction_output_dir, textdescs_folder)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTranslation completed in {elapsed_time:.2f} seconds.")
    else:
        print("The specified input folder does not exist.")


def main():
    dataset='sap_sam_2022/filtered'
    prediction_output_dir = f'data/evaluation/{dataset}/test/vanilla_'

    #load train cases
    dataset_for_training_dir = f'data/{dataset}/forTraining'#'data/bpmai/forTraining_flan_t5'
    training_dataset_filename = 'training'
    path_to_training_dataset = os.path.join(dataset_for_training_dir, training_dataset_filename)
    data = load_from_disk(path_to_training_dataset)
    validation_data = data['test'].to_pandas()
    model_case_names = validation_data.id.unique() 

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process event logs with optional text descriptions.")
    parser.add_argument("--use_textdescs", action="store_true", help="Use text descriptions folder during processing.")
    parser.add_argument("--llama", action="store_true", help="Use Llama instead of GPT-4o.")
    parser.add_argument("--everything", action="store_true", help="Run all combinations of models and with/without textdesc")

    args = parser.parse_args()

    if args.everything:
        # Run all combinations
        for llm in ["gpt-4o", "meta--llama3.1-70b-instruct"]:
            for use_textdescs in [False, True]:
                run_model(llm, use_textdescs, dataset, model_case_names)
    else:
        # Choose model based on argument
        if args.llama:
            llm = "meta--llama3.1-70b-instruct"
        else:
            llm = "gpt-4o"
        
        # Run based on text description argument
        use_textdescs = args.use_textdescs
        run_model(llm, use_textdescs, dataset, model_case_names)


if __name__ == "__main__":
    main()


"""
dataset='sap_sam_2022/filtered'
prediction_output_dir = f'data/evaluation/{dataset}/test/vanilla_'

#load train cases
dataset_for_training_dir = f'data/{dataset}/forTraining'#'data/bpmai/forTraining_flan_t5'
training_dataset_filename = 'training'
path_to_training_dataset = os.path.join(dataset_for_training_dir, training_dataset_filename)
data = load_from_disk(path_to_training_dataset)
validation_data = data['test'].to_pandas()
model_case_names = validation_data.id.unique() 

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process event logs with optional text descriptions.")
    parser.add_argument("--use_textdescs", action="store_true", help="Use text descriptions folder during processing.")
    parser.add_argument("--llama", action="store_true", help="Use Llama instead of GPT-4o.")
    parser.add_argument("--everything", action="store_true", help="Run all combinations of models and with/without textdesc")

    args = parser.parse_args()
    
    # Choose model based on argument, standard gpt-4o
    if args.llama:
        llm = "meta--llama3.1-70b-instruct"
    else:
        llm = "gpt-4o"

    prediction_output_dir = prediction_output_dir + llm
    # Input and output folder paths
    eventlogs_folder = f'data/{dataset}/logs'
    textdescs_folder = None
    # Validate textdescs_folder if --use_textdescs is set
    if args.use_textdescs:
        textdescs_folder = f'data/{dataset}/textdescs'
        prediction_output_dir = prediction_output_dir + "_combined/"
        if not os.path.exists(textdescs_folder):
            print("Error: The text descriptions folder does not exist.")
            print("You need to run declare_to_textdesc.py file before this to generate the textdescs")
            exit(1)
    else:
        prediction_output_dir = prediction_output_dir + "_eventlog/"

    if os.path.exists(eventlogs_folder):
        # Start timing
        start_time = time.time()

        # Process files
        process_files(llm, model_case_names, eventlogs_folder,  prediction_output_dir, textdescs_folder)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTranslation completed in {elapsed_time:.2f} seconds.")
    else:
        print("The specified input folder does not exist.")
"""