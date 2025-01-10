import argparse
from tqdm import tqdm
from evaluation.utils import evaluate_constraints
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser(description="Process and print eval results with optional settings.")
parser.add_argument("--llama", action="store_true", help="Print results only of Llama models")
parser.add_argument("--gpt-4o", action="store_true", help="Print results only of gpt-4o models")
parser.add_argument("--random", action="store_true", help="Print results only of random predictions")
parser.add_argument("--kiran", action="store_true", help="Print results only of Kiran's predictions")
parser.add_argument("--grouped", action="store_true", help="Eval only on grouped constraints")
parser.add_argument("--eventlog", action="store_true", help="Evaluate only eventlog models")
parser.add_argument("--combined", action="store_true", help="Evaluate only combined models")

args = parser.parse_args()

# Define variables
model_checkpoint = "checkpoint-12276"
model_name = 'google/flan-t5-small'
dataset = 'sap_sam_2022/filtered'
base_dir = f'data/evaluation/{dataset}/'

path_to_true_constraints = 'data/sap_sam_2022/filtered/constraints_to_log_labels'
constraints_to_be_grouped = [
    'Response', 'Precedence', 'Succession', 'Alternate Succession', 
    'Alternate Precedence', 'Alternate Response', 'Choice', 'Co-Existence'
]

if not args.grouped:
    constraints_to_be_grouped = None

# Function to dynamically discover models
def get_model_dirs(base_dir):
    models = []
    test_case_names = []
    
    for root, dirs, files in os.walk(base_dir):
        for test_dir in dirs:
            test_path = os.path.join(root, test_dir)
            for sub_dir in os.listdir(test_path):
                sub_dir_path = os.path.join(test_path, sub_dir)
                if os.path.isdir(sub_dir_path):
                    # Extract test case names only once
                    if not test_case_names:
                        test_case_names = [
                            i.split('.')[0] for i in os.listdir(sub_dir_path) if i.endswith('.pkl')
                        ]

                        # Not interessted with cases that has no constraints
                        if len(test_case_names) == 0:
                            continue
                    
                    # Apply filters based on args
                    if args.llama and "llama" in sub_dir:
                        if args.eventlog and "eventlog" in sub_dir:
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                        elif args.combined and "combined" in sub_dir:
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                        elif not (args.eventlog or args.combined):  # If neither flag is passed, include all llama models
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                    
                    elif args.gpt_4o and "gpt-4o" in sub_dir:
                        if args.eventlog and "eventlog" in sub_dir:
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                        elif args.combined and "combined" in sub_dir:
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                        elif not (args.eventlog or args.combined):  # If neither flag is passed, include all gpt-4o models
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                    
                    elif args.random and "random" in sub_dir:
                        models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
                    
                    elif args.kiran and "flan-t5-small_checkpoint" in sub_dir_path:
                        models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, 0.69))
                    
                    elif not (args.llama or args.gpt_4o or args.random or args.kiran):  # If no filter, include all
                        if "flan-t5-small_checkpoint" in sub_dir_path:
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, 0.69))
                        else:
                            models.append((test_case_names, path_to_true_constraints, sub_dir_path, sub_dir, constraints_to_be_grouped, None, None))
    return models

# Collect models
models = get_model_dirs(base_dir)

# Dictionary to store results
results_by_model_type = {}

# Evaluate constraints
for model in models:
    print('----- ', model[3], ' -----')
    
    # Skip missing or empty folders
    if not os.path.exists(model[2]) or len(os.listdir(model[2])) == 1:
        print(f"Skipping missing or empty folder: {model[2]}")
        continue
    
    result = evaluate_constraints(*model[:])
    output_file_name = f'{model[3]}.pkl'
    df = pd.DataFrame(result)
    df.to_pickle(output_file_name)
    
    # Add results to dictionary for aggregation
    model_type = model[3]
    if model_type not in results_by_model_type:
        results_by_model_type[model_type] = []
    results_by_model_type[model_type].append(df)
    
    if model[4]:
        print(pd.read_pickle(output_file_name)[['precision', 'recall', 'f1']].mean().round(2))
    else:
        print(pd.read_pickle(output_file_name)[['constraint_type', 'precision', 'recall', 'f1']]
              .groupby(['constraint_type']).mean().reset_index().round(2))


# Function to generate clean LaTeX tables
def _generate_clean_latex_table(df, model_type, output_dir):
    latex_table = f"""
\\begin{{table}}[H]
    \\centering
    \\caption{{Performance Comparison for \\texttt{{{model_type}}}}}
    \\label{{tab:{model_type.replace('_', '')}}}
    \\begin{{tabular}}{{lccc}}
        \\hline
        \\textbf{{Constraint Type}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} \\\\ \\hline
"""
    for _, row in df.iterrows():
        latex_table += f"        {row['constraint_type']} & {row['precision']:.2f} & {row['recall']:.2f} & {row['f1']:.2f} \\\\\n"
    
    latex_table += """        \\hline
    \\end{tabular}
\\end{table}"""

    # Save to file
    file_path = os.path.join(output_dir, f"{model_type}_clean_table.tex")
    with open(file_path, "w") as file:
        file.write(latex_table)
    print(f"Clean LaTeX table saved to: {file_path}")


# Aggregate results
print("\nAggregated Results Across Test Folders:\n")

# Directory to save results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Combine, print, save tables, and generate combined plots
metrics = ['precision', 'recall', 'f1']
combined_results = {}  # Store all model results for combined plotting

for model_type, dfs in results_by_model_type.items():
    # Combine results for the current model
    combined_df = pd.concat(dfs, ignore_index=True)
    mean_results = combined_df[['constraint_type', 'precision', 'recall', 'f1']] \
        .groupby(['constraint_type']).mean().reset_index().round(2)
    
    # Print table in console
    print(f"Mean results for model type '{model_type}':\n", mean_results)
    
    # Save LaTeX table
    _generate_clean_latex_table(mean_results, model_type, output_dir)
    
    # Store results for combined plots
    combined_results[model_type] = mean_results

# Generate combined bar plots for all models
for metric in metrics:
    plt.figure(figsize=(12, 7))
    
    # Prepare x-axis positions
    constraint_types = combined_results[next(iter(combined_results))]['constraint_type']
    x = np.arange(len(constraint_types))  # Group positions on x-axis
    
    bar_width = 0.15  # Width of each bar
    offset = 0  # Track position for each model
    
    for model_type, mean_results in combined_results.items():
        plt.bar(x + offset, mean_results[metric], width=bar_width, label=model_type)
        offset += bar_width  # Shift bars for next model
    
    # Plot settings
    plt.xticks(x + bar_width * (len(combined_results) / 2 - 0.5), constraint_types, rotation=45, ha="right")
    plt.title(f'Comparison of {metric.capitalize()} Across Models')
    plt.xlabel('Constraint Type')
    plt.ylabel(metric.capitalize())
    plt.legend(title="Model Type", loc='upper left')
    plt.tight_layout()
    
    # Save and display figure
    figure_path = os.path.join(output_dir, f"combined_{metric}.png")
    plt.savefig(figure_path)
    print(f"Combined {metric} bar plot saved to: {figure_path}")
    plt.show()