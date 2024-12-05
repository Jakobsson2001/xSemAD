from more_itertools import chunked
import numpy as np
from labelparser.label_utils import constraint_splitter, sanitize_label
import os
import json
import pickle
from tqdm import tqdm
import re

def _prediction_cleaning(p):
    p = p.replace("<pad>","").replace("</s>","")
    p = p.strip()
    p = " ".join(p.split()) 
    return p

def _collect_scores(predictions,scores):
    predictions_dict = dict()
    maxLen = 1
    for p,s in zip(predictions,scores):
        if p in predictions_dict:
            predictions_dict[p].append(s)
            if len(predictions_dict[p]) > maxLen:
                maxLen = len(predictions_dict[p])
        if p not in predictions_dict:
            predictions_dict[p] = [float(s)]
    return predictions_dict, maxLen

def __ranking_max(predictions, scores,num_recommondations):
    predictions_dict, maxLen = _collect_scores(predictions,scores)
    # save the activities together with their confidences in a list but such that every activity has the same number of confidences (add 0's)
    # also sort the confidences per activity
    predictions_list = []
    if maxLen == 1:
        predictions_list = [(k,v) for k,v in predictions_dict.items()]
    else:
        for k,v in predictions_dict.items():
            v.sort(reverse=True)
            while len(v) < maxLen:
                v.append(0)
            predictions_list.append((k,v))
    # now sort the activites according to their confidences
    ranking = sorted(predictions_list, key=lambda tup: tuple(map(lambda i: tup[1][i],list(range(len(tup[1]))))), reverse=True)
    # if two or more activites have the same confidence, sort them according to their label
    ranking = sorted(ranking, key=lambda tup: (tup[1][0],tup[0]), reverse=True)
    # reduce the sorted list to the top10 activities with their maximum confidences
    return ranking[:num_recommondations]


def sort_constraints(constraints_list,correct_spelling=False, remove_duplicates=True):
    result=[]
    for c in constraints_list:
        constraint_type, labels = constraint_splitter(c, correct_spelling=correct_spelling)
        if labels == None:
            return result
        labels = [i.strip().replace('  ',' ').lower() for i in labels]
        if constraint_type.lower() in ['coice', 'co-existence','exclusive choice']:
            labels.sort()
        result.append(f'{constraint_type}['+ ', '.join(labels) +']')
    if remove_duplicates:
        result = list(set(result))
    return result


def generate_prediction_list(input_sequences, tokenizer, model, num_recommendations, max_new_tokens=200, device='cpu'):
    """
    Generate a list of predictions for a given input sequence using a sequence-to-sequence language model, 
    along with associated confidence scores based on sequence probabilities.

    Parameters:
    - input_sequences (str or list of str): The input text(s) for which predictions are generated. 
      Can be a single string or a list of strings.
    - tokenizer: The tokenizer associated with the model, used to encode input text and decode predictions.
    - model: The pre-trained sequence-to-sequence language model (e.g., loaded with AutoModelForSeq2SeqLM) to 
      generate predictions.
    - num_recommendations (int): The number of recommended sequences to generate per input sequence.
    - max_new_tokens (int, optional): Maximum number of new tokens to generate per sequence. Defaults to 200.
    - device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
    - recommendations_with_score (list of tuples): A list of tuples, where each tuple contains:
        - The generated prediction (str) for the input sequence.
        - The associated probability score (float), rounded to three decimal places, representing the 
          model’s confidence in the prediction. Scores are converted from log-probabilities.

    Notes:
    - Log-probabilities returned by the model for each sequence are converted to probabilities using `np.exp`.
    - High values for `num_recommendations` or `max_new_tokens` may slow down performance.
    - The `no_repeat_ngram_size` parameter is set to avoid repeated n-grams within each sequence.
    """

    # Tokenize input sequences for the model and move tensors to the specified device
    inputs = tokenizer(input_sequences, return_tensors='pt', padding=True).to(device)
    
    # Generate sequences with specified generation parameters
    sample_output = model.generate(
        max_new_tokens=max_new_tokens,
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        num_return_sequences=num_recommendations,
        num_beams=num_recommendations,
        no_repeat_ngram_size=50,  # Prevent repetition of n-grams within each generated sequence
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    predictions, scores = [], []
    
    # Process each generated sequence and its associated score
    for preds_sequence, scores_sequence, sequence in zip(
            chunked(sample_output["sequences"].cpu(), num_recommendations),
            chunked(sample_output["sequences_scores"].cpu(), num_recommendations),
            input_sequences):
        
        # Decode the generated sequences back to text and clean up each prediction
        preds_sequence = tokenizer.batch_decode(preds_sequence, skip_special_tokens=True)
        preds_sequence = [_prediction_cleaning(p) for p in preds_sequence]
        
        # Extend the list of predictions and scores
        predictions.extend(preds_sequence)
        scores.extend(scores_sequence)
    
    # Convert log-probability scores to probabilities
    scores = np.exp(scores)  # Converts log-probabilities to probabilities in the range [0, 1]
    
    # Rank predictions based on scores, select the top `num_recommendations`, and round scores to three decimals
    recommendations_with_score = [
        (r, round(float(s[0]), 3)) for r, s in __ranking_max(predictions, scores, num_recommendations)
    ]

    # Calculate and print the mean score for insight
    mean_score = np.mean([s[1] for s in recommendations_with_score])
    print(f"Mean confidence score for generated predictions: {mean_score:.3f}")
    
    return recommendations_with_score


def filter_prediction_list_for_eval(model_labels,prediction_c_list):
    result=[]
    for prediction in prediction_c_list:
        c, labels = constraint_splitter(prediction[0], correct_spelling=False)
        if labels != None:
            labels = [i.strip() for i in labels]
            if set(labels).issubset(model_labels):
                result.append((prediction[0],prediction[1]))
    return result


def sort_constraints_for_eval(constraints_list,correct_spelling=False, remove_duplicates=True):
    result=[]
    for c in constraints_list:
        constraint_type, labels = constraint_splitter(c[0], correct_spelling=correct_spelling)
        if labels == None:
            return result
        labels = [i.strip() for i in labels]
        if constraint_type.lower() in ['coice', 'co-existence','exclusive choice']:
            labels.sort()
        result.append((f'{constraint_type}['+ ', '.join(labels) +']', c[1]))
    if remove_duplicates:
        result = list(set(result))
    return result


def _calculate_precision_recall_f1(true_list, prediction_list):
    """
    Calculate precision, recall, and F1 score based on true and predicted lists of items.
    
    Parameters:
    - true_list (list): The list of actual relevant items (ground truth), 
      representing the sum of true positives and false negatives.
    - prediction_list (list): The list of predicted items, which may overlap with true_list.

    Returns:
    - tuple: A tuple (precision, recall, f1) where:
        - precision (float): The ratio of correctly predicted items to the total predicted items.
        - recall (float): The ratio of correctly predicted items to the total actual items in true_list.
        - f1 (float): The harmonic mean of precision and recall.
      If `true_list` is empty (no items to evaluate), or if both `true_list` and `prediction_list` are empty,
      returns (None, None, None).
    
    Notes:
    - Precision is set to 0 if `prediction_list` is empty, provided `true_list` is non-empty.
    - Recall and F1 are only calculated if `true_list` is non-empty.
    """
    
    # Return None for all scores if there are no true items to evaluate
    if len(true_list) == 0:
        return None, None, None
    
    # Calculate the number of correctly predicted items (True Positives) by finding the intersection
    # between the sets of true and predicted items, then taking the length of that intersection.
    intersection_num = len(set(true_list).intersection(set(prediction_list)))
    
    # Calculate recall
    recall = intersection_num / len(true_list)
    
    # Calculate precision
    precision = intersection_num / len(prediction_list) if len(prediction_list) > 0 else 0
    
    # Calculate F1 score
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    return precision, recall, f1


def evaluate_constraints(test_case_names, 
                         path_to_true_constraints, 
                         path_to_pred_constraints,
                         MODEL_NAME="Unsett",  # Default MODEL_NAME
                         group_constraint_types=None,
                         unseen_model_case_names=None,
                         xsemad_threshold=0.68, # Default threshold, can be adjusted
                         constraints_of_interest = ['Alternate Precedence',
                                                    'Alternate Response',
                                                    'Alternate Succession',
                                                    'Choice',
                                                    'Co-Existence',
                                                    'End',
                                                    'Exclusive Choice',
                                                    'Init',
                                                    'Precedence',
                                                    'Response',
                                                    'Succession']):
    """
    Evaluate the performance of a model on various constraint types by calculating precision, 
    recall, and F1 scores for each specified constraint across a set of test cases.

    Parameters:
    - test_case_names (list of str): List of test case names to evaluate.
    - path_to_true_constraints (str): Path to the directory containing the ground truth constraints 
      for each test case.
    - path_to_pred_constraints (str): Path to the directory containing the model's predicted 
      constraints for each test case.
    - MODEL_NAME (str, optional): Name of the model, used to categorize results. Defaults to "Unsett".
    - group_constraint_types (list of str, optional): List of constraint types to group together 
      for a combined evaluation. If provided, precision, recall, and F1 are calculated for the 
      grouped constraints as a whole.
    - unseen_model_case_names (list of str, optional): Not sure if needed for this implementationa and
      what this is for but my best guess is that it is a subset of `test_case_names` representing 
      cases that were not seen during training. If provided, the evaluation will only consider 
      cases within this list, allowing for a separate evaluation on "unseen" data to assess 
      generalization performance.
    - xsemad_threshold (float, optional): Threshold value for filtering predictions based on 
      confidence or relevance scores. Only predictions above this threshold are considered 
      in the evaluation.
    - constraints_of_interest (list of str, optional): List of constraint types to evaluate. 
      Only these constraints are considered in both the true and predicted constraints.
      
    Returns:
    - evaluation_results (list of dict): A list where each dictionary contains the evaluation 
      metrics (precision, recall, F1 score) for a constraint type or grouped constraint. Each 
      dictionary includes:
        - 'constraint_type': The constraint type or grouped constraint types being evaluated.
        - 'model': The model name.
        - 'precision': The precision score for the constraint(s).
        - 'recall': The recall score for the constraint(s).
        - 'f1': The F1 score for the constraint(s).
        - 'case_name': The name of the test case.
    
    Notes:
    - If both `group_constraint_types` and `constraints_of_interest` are provided, only the 
      constraints specified in `constraints_of_interest` that match the `group_constraint_types` 
      will be grouped and evaluated together.
    - If `unseen_model_case_names` is provided, `test_case_names` will be filtered to only include 
      cases in both `test_case_names` and `unseen_model_case_names`.
    """

    evaluation_results = []
    
    if unseen_model_case_names:
        test_case_names = [name for name in unseen_model_case_names if name in test_case_names]
    
    for model_case_name in tqdm(test_case_names, desc='Processing evaluations'):
        true_constraints = _load_constraints(path_to_true_constraints, model_case_name)
        all_constraint_types_in_model = _extract_constraint_types(true_constraints)
        
        pred_pairs_temp = _load_predictions(path_to_pred_constraints, model_case_name, xsemad_threshold)

        if group_constraint_types is not None:
            # Perform grouped evaluation and get results
            result = _grouped_evaluation(true_constraints, pred_pairs_temp, group_constraint_types, constraints_of_interest, model_case_name, MODEL_NAME)
            if result:
                evaluation_results.append(result)
        else:
            # Perform individual evaluation and get results
            individual_results = _individual_evaluation(true_constraints, pred_pairs_temp, all_constraint_types_in_model, constraints_of_interest, model_case_name, MODEL_NAME)
            evaluation_results.extend(individual_results)  # Add all individual results to the main results list
    
    return evaluation_results


def _load_constraints(path, model_case_name):
    """Load and sort true constraints for a specific test case."""
    with open(f'{path}/{model_case_name}.CONSTRAINTS.pkl', 'rb') as f:
        true_constraints = pickle.load(f)
    return sort_constraints(true_constraints, remove_duplicates=True)


def _extract_constraint_types(constraints):
    """Extract unique constraint types from a list of constraints."""
    return list(set([constraint.split('[')[0] for constraint in constraints]))


def _load_predictions(path, model_case_name, threshold):
    """Load prediction constraints, applying a threshold filter to get only generated constraints with good enough ranking TODO: is this properly explained?."""
    path_to_pred_file = f'{path}/{model_case_name}.pkl'

    try:
        with open(path_to_pred_file, 'rb') as f:
            pred_pairs_temp = pickle.load(f)
    except (FileNotFoundError, IOError) as e:
        # Handle the error and continue to the next iteration
        print(f"Could not find file {path_to_pred_file}: {e}")
        exit()
            

    if threshold is not None: 
        pred_pairs_temp = [item for sublist in pred_pairs_temp for item in sublist[1]]
        pred_pairs_temp = [i for i in pred_pairs_temp if i[1] > threshold]
        return sort_constraints([i[0] for i in pred_pairs_temp], remove_duplicates=True)
    #print(pred_pairs_temp[0])
    # Structure adjustment for XSEMAD
    return sort_constraints(pred_pairs_temp, remove_duplicates=True)


def _grouped_evaluation(true_constraints, pred_constraints, group_types, constraints_of_interest, case_name, model_name):
    """Evaluate grouped constraint types and return result."""
    true_pairs, pred_pairs = [], []
    for constraint_type in constraints_of_interest:
        if constraint_type in group_types:
            true_pairs += _extract_pairs(true_constraints, constraint_type)
            pred_pairs += _extract_pairs(pred_constraints, constraint_type)
    
    # Evaluate if there are true pairs
    if true_pairs:
        precision, recall, f1 = _calculate_precision_recall_f1(list(set(true_pairs)), list(set(pred_pairs)))
        return {
            'constraint_type': ', '.join(group_types),
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'case_name': case_name
        }
    return None  # Return None if no true pairs were found


def _individual_evaluation(true_constraints, pred_constraints, constraint_types_in_model, constraints_of_interest, case_name, model_name):
    """Evaluate each constraint type individually and return a list of results."""
    results = []
    zero_count = 0
    for constraint_type in constraints_of_interest:
        if constraint_type in constraint_types_in_model:
            true_pairs = _extract_pairs(true_constraints, constraint_type)
            pred_pairs = _extract_pairs(pred_constraints, constraint_type)

            if true_pairs:
                precision, recall, f1 = _calculate_precision_recall_f1(list(set(true_pairs)), list(set(pred_pairs)))
                results.append({
                    'constraint_type': constraint_type,
                    'model': model_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'case_name': case_name
                })
                
                if recall == 0 and precision == 0:
                    zero_count += 1

    #print(f"Zero-score constraints for {case_name}: {zero_count} out of {len(constraints_of_interest)}")
    return results  # Return the list of results for each individual constraint


def _extract_pairs(constraints, constraint_type):
    """Extract constraint pairs that match a given type."""
    return [constraint.split('[')[1][:-1] for constraint in constraints if constraint.startswith(constraint_type + '[')]
