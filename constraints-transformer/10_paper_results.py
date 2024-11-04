{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pickle\n",
    "import os\n",
    "from glob import glob\n",
    "\n",
    "model_checkpoint = \"checkpoint-12276\"\n",
    "model_name='google/flan-t5-small'\n",
    "dataset='sap_sam_2022/filtered'\n",
    "# Define the path to the output directory where .pkl files for test scores are stored\n",
    "prediction_output_dir = f'data/evaluation/{dataset}/test/{model_name}_{model_checkpoint}/'\n",
    "\n",
    "# Initialize an empty DataFrame to accumulate results from all files\n",
    "all_results = []\n",
    "\n",
    "# Loop through all .pkl files in the directory\n",
    "for file_path in glob(os.path.join(prediction_output_dir, '*.pkl')):\n",
    "    # Load each .pkl file\n",
    "    with open(file_path, 'rb') as f:\n",
    "        results = pickle.load(f)\n",
    "    \n",
    "    # Convert each result into a DataFrame with the expected structure\n",
    "    for constraint_type, predictions in results:\n",
    "        for pred in predictions:\n",
    "            all_results.append({\n",
    "                'constraint_type': constraint_type,\n",
    "                'precision': pred.get('precision', 0),\n",
    "                'recall': pred.get('recall', 0),\n",
    "                'f1': pred.get('f1', 0)\n",
    "            })\n",
    "\n",
    "# Convert the accumulated list into a DataFrame\n",
    "df = pd.DataFrame(all_results)\n",
    "\n",
    "# Group by 'constraint_type' and calculate mean precision, recall, and f1, rounded to 2 decimals\n",
    "evaluation_scores = df.groupby('constraint_type').mean().reset_index().round(2)\n",
    "\n",
    "# Display the results\n",
    "print(evaluation_scores)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "declare_miner",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
