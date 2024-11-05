from tqdm import tqdm
from evaluation.utils import evaluate_constraints
import pickle
import os
import pandas as pd

model_checkpoint = "checkpoint-12276"
model_name='google/flan-t5-small'
dataset='sap_sam_2022/filtered'
# Define the path to the output directory where .pkl files for test scores are stored
prediction_output_dir = f'data/evaluation/{dataset}/test/{model_name}_{model_checkpoint}/'
test_case_names = [i.split('.')[0] for i in os.listdir(prediction_output_dir)]

path_to_true_constraints='data/sap_sam_2022/filtered/constraints_to_log_labels'
constraints_to_be_grouped = ['Response', 'Precedence','Succession','Alternate Succession', 'Alternate Precedence','Alternate Response','Choice','Co-Existence']

models= [
    #(test_case_names, path_to_true_constraints, prediction_output_dir, 'xSemAD_evf', constraints_to_be_grouped, None, 0.73),
    (test_case_names, path_to_true_constraints, prediction_output_dir, 'xSemAD_seperated', None, None,0.73),
    # 100% UNSEEN DATA
    #(test_case_names, path_to_true_constraints, prediction_output_dir, 'xSemAD_evf_unseen', constraints_to_be_grouped, unseen_case_names, 0.73),
    #(test_case_names, path_to_true_constraints, prediction_output_dir, 'xSemAD_evf_unseen_065', constraints_to_be_grouped, unseen_case_names, 0.65),
    #(test_case_names, path_to_true_constraints, prediction_output_dir, 'xSemAD_seperated_unseen_065', None, unseen_case_names, 0.65),
]

for model in models:
    print('----- ', model[3],' -----')
    result =  evaluate_constraints(*model[:])
    output_file_name= f'{model[-1]}_{model[3]}.pkl'
    df = pd.DataFrame(result)
    df.to_pickle(output_file_name)
    if model[4]:
        print(pd.read_pickle(output_file_name)[['precision','recall','f1']].mean().round(2))
    else:
        print(pd.read_pickle(output_file_name)[['constraint_type', 'precision', 'recall','f1']].groupby(['constraint_type']).mean().reset_index().round(2))