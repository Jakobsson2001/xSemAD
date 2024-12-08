import os
import pickle
import json
import time
from dotenv import load_dotenv

from configGenAI import init

# Load environment variables from .env file
load_dotenv()

openai = init()

def describe_with_openai(declare_content):
    """
    Uses GPT-4o via OpenAI API (through GenAIHubProxyClient) to generate descriptions for DECLARE content.
    """
    try:
        """
        # Second suggestion of prompt in teamschat
        messages = [
            {"role": "system", "content": "You are an expert in translating process mining rules into plain text."},
            {"role": "user", "content": f"Translate the following DECLARE rules directly into readable text descriptions without any introductory text: {declare_content}"}
        ]
        """
        # Combine all constraints into a single prompt (for last two prompt suggestions)
        all_constraints = "\n".join(declare_content)
        #print(all_constraints)
        """
        # Example three in teams convo, more fluant desc of all declare rules
        messages = [
            {"role": "system", "content": "You are an expert in process mining and translating rules into natural language."},
            {"role": "user", "content": (
                "Here are DECLARE rules describing a process. Generate a single cohesive textual description of the entire process, "
                "explaining its key activities, their relationships, and the overall flow:\n"
                f"{all_constraints}"
            )}
            #{"role": "user", "content": f"Translate the following DECLARE rules directly into readable text descriptions without any introductory text: {declare_content}"}

        ]
        """
        # Last example prompt in teams chat
        messages = [
            {"role": "system", "content": "You are an expert in process mining and translating rules into natural language summaries."},
            {"role": "user", "content": (
                "The following DECLARE rules describe a business process. Please generate a natural, prosaic summary of this process. "
                "Focus on creating a flowing narrative that explains the sequence of events, the decisions involved, and the overall flow of the process. "
                "Avoid listing the rules; instead, weave them into a cohesive explanation:\n"
                f"{all_constraints}"
            )}
        ]

        response = openai.chat.completions.create(
            model_name="gpt-4o",
            messages=messages,
            temperature=0.5,
            max_tokens=1000,  # Adjust token limit as needed
            frequency_penalty=0,
            top_p=1,
            seed=1234
        )
    
        # Access the message content properly
        out = json.loads(response.json())
        result = out["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        return f"Error generating description: {e}"

def process_declare_files(relevant_names_folder, input_folder, output_folder):
    """
    Processes all DECLARE .pkl files in the input folder and saves descriptions as .txt files in the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    counter = 0 
    nr_files = len(os.listdir(relevant_names_folder))/2
    for file_name in os.listdir(relevant_names_folder):
        if file_name.endswith("CONSTRAINTS.pkl"):
            counter += 1
            case_name = file_name.split('.')[0]
            file_path = os.path.join(input_folder, case_name + ".DECLARE.pkl")
            print(f"Processing file: {file_name}. Nr: {counter}/{nr_files}")
            try:
                with open(file_path, 'rb') as file:
                    declare_content = pickle.load(file)
                
                if isinstance(declare_content, list):  # Ensure the content is a list of rules
                    description = describe_with_openai(declare_content)
                    
                    # Save the description to a .txt file with the same name as the .pkl file
                    output_file_name = case_name + ".txt"
                    output_file_path = os.path.join(output_folder, output_file_name)
                    
                    with open(output_file_path, 'w') as output_file:
                        output_file.write(description)
                    
                    """
                    print(f"Description saved to {output_file_path}")
                    print("------Declare content------")
                    print(declare_content)
                    print("---------Text desc------------")
                    print(description)
                    exit()
                    """
                else:
                    print(f"Unexpected content in {file_name}: {type(declare_content)}")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

if __name__ == "__main__":
    # Input and output folder paths
    relevant_names_folder = "data/sap_sam_2022/filtered/constraints_to_log_labels/"
    input_folder = "data/sap_sam_2022/filtered/constraints/"
    output_folder = "data/sap_sam_2022/filtered/textdescs/"

    if os.path.exists(input_folder) and os.path.exists(relevant_names_folder):
        # Start timing
        start_time = time.time()
        
        # Process files
        process_declare_files(relevant_names_folder, input_folder, output_folder)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTranslation completed in {elapsed_time:.2f} seconds.")
    else:
        print("The specified input folder does not exist.")
