from more_itertools import chunked
import numpy as np
from labelparser.label_utils import constraint_splitter, sanitize_label
import os
import json
import pickle
from tqdm import tqdm
import re

def prediction_cleaning(p):
    p = p.replace("<pad>","").replace("</s>","")
    p = p.strip()
    p = " ".join(p.split()) 
    return p

def collect_scores(predictions,scores):
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
    predictions_dict, maxLen = collect_scores(predictions,scores)
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

def generate_prediction_list(input_sequences, tokenizer, model, num_recommondations, max_new_tokens=200, device='cpu'):
    inputs = tokenizer(input_sequences,return_tensors='pt',padding=True).to(device)
    sample_output = model.generate(
                max_new_tokens = max_new_tokens,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_return_sequences=num_recommondations,
                num_beams=num_recommondations,
                no_repeat_ngram_size = 50, # no word repetitions
                early_stopping=True, #True,
                return_dict_in_generate=True,
                output_scores=True
                )
    predictions=[]
    scores=[]
    for preds_sequence,scores_sequence,sequence in zip(chunked(sample_output["sequences"].cpu(),num_recommondations),
                                                       chunked(sample_output["sequences_scores"].cpu(),num_recommondations),input_sequences):
        preds_sequence = tokenizer.batch_decode(preds_sequence, skip_special_tokens=False)
        preds_sequence = [prediction_cleaning(p) for p in preds_sequence]
        #preds_sequence = list(set([p for p in preds_sequence if p!=""]))[:num_recommondations]
        #if len(preds_sequence)<10:
        #   print("length of rec list"+str(len(preds_sequence)))
        predictions.extend(preds_sequence)
        scores.extend(scores_sequence)
    scores = np.exp(scores)#softmax(scores)#np.exp(scores)#1/(1+np.exp([-s for s in scores])) #np.exp(scores)
    recommendations_with_score = [(r,round(float(s[0]),3)) for r,s in __ranking_max(predictions, scores,num_recommondations)]
    return recommendations_with_score


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

def filter_prediction_list(model_labels,prediction_c_list):
    result=[]
    for prediction in prediction_c_list:
        c, labels = constraint_splitter(prediction, correct_spelling=False)
        if labels != None:
            labels = [i.strip() for i in labels]
            if set(labels).issubset(model_labels):
                result.append(prediction)
    return result

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


def calculate_precision_recall_f1(true_list, prediction_list):
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
                         MODEL_NAME=None,
                         group_constraint_types=None,
                         unseen_model_case_names=None,
                         xsemad_threshold=0.68, # Find optimal threshold, insert here as standard
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

    evaluation_results = []
    model_type = MODEL_NAME.split('_')[0]

    pred_file_type = os.listdir(path_to_pred_constraints)[0].split('.')[1]

    if unseen_model_case_names:
        test_case_names=[item for item in unseen_model_case_names if item in test_case_names]
    for model_case_name in tqdm(test_case_names, desc='process evaluation'):
        # Load true constraints
        with open(f'{path_to_true_constraints}/{model_case_name}.CONSTRAINTS.pkl','rb') as f:
            true_constraints = pickle.load(f)
            all_constraint_types_in_model = list(set([i.split('[')[0] for i in true_constraints]))
            true_constraints = sort_constraints(true_constraints, remove_duplicates=True)
        
        # Load predictions
        # PREDICTION
        path_to_pred_file = f'{path_to_pred_constraints}/{model_case_name}.{pred_file_type}'
        
        if pred_file_type in ['pkl','pickle']:
            with open(path_to_pred_file, 'rb') as f:
                pred_pairs_temp = pickle.load(f)#pred_pairs_temp = sort_constraints(pickle.load(f), remove_duplicates=True)
                # For XSEMAD, filter predictions based on the threshold
                pred_pairs_temp = [item for sublist in pred_pairs_temp for item in sublist[1]]
                pred_pairs_temp = [i for i in pred_pairs_temp if i[1] > xsemad_threshold]  # Apply threshold filtering

            # pred_pairs_temp structure adjustment for XSEMAD predictions is needed
            pred_pairs_temp = sort_constraints([i[0] for i in pred_pairs_temp], remove_duplicates=True)

        if group_constraint_types is not None:
            true_pairs=[]
            pred_pairs=[]
            for c in constraints_of_interest:
                if c in group_constraint_types:
                    print(c)
                    true_pairs_ = [i.split('[')[1][:-1] for i in true_constraints if i.startswith(c+ '[') ]
                    print(true_pairs_)
                    true_pairs+=true_pairs_
                    pred_pairs_ = [i.split('[')[1][:-1] for i in pred_pairs_temp if i.startswith(c+ '[')] 
                    pred_pairs+=pred_pairs_
                    print("\nPred:",pred_pairs_, "\n")
            if len(true_pairs)>0:
                precision, recall, f1 = calculate_precision_recall_f1(true_list=list(set(true_pairs)), prediction_list=list(set(pred_pairs)))
                evaluation_results.append({'constraint_type':', '.join(group_constraint_types), 'model':MODEL_NAME, 'precision':precision,'recall':recall,'f1':f1, 'case_name':model_case_name})
        else:
            nrZeroes = 0
            for c in constraints_of_interest:
                true_pairs=[]
                pred_pairs=[]
                if c in all_constraint_types_in_model:
                    true_pairs = [i.split('[')[1][:-1] for i in true_constraints if i.startswith(c+ '[') ]
                    pred_pairs = [i.split('[')[1][:-1] for i in pred_pairs_temp if i.startswith(c+ '[')] 
                    if len(true_pairs)>0:
                        print(len(pred_pairs), " : ",len(true_pairs))
                        precision, recall, f1 = calculate_precision_recall_f1(true_list=list(set(true_pairs)), prediction_list=list(set(pred_pairs)))
                        evaluation_results.append({'constraint_type':c, 'model':MODEL_NAME, 'precision':precision,'recall':recall,'f1':f1, 'case_name':model_case_name})
                        if(recall == 0 and precision == 0):
                            nrZeroes+=1
                            print(c)
                        #print(evaluation_results)

            print(nrZeroes, "?=")
            print(len(constraints_of_interest))
    return evaluation_results