#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

cd constraints-transformer/

# Install BLEURT from GitHub
pip3 install git+https://github.com/google-research/bleurt.git

# Install transformer4bpm
pip3 install git+https://github.com/disola/bpart5.git

pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz --no-deps

# Clone the bpmn2constraints repository if not already cloned
if [ ! -d "bpmn2constraints" ]; then
    git clone https://github.com/signavio/bpmn2constraints.git
fi

# Navigate into the repository directory
cd bpmn2constraints

# Install the package using pip
pip3 install .

# Install the rest of the requirements
pip3 install -r requirements.txt

# Create the directory structure if it doesn't exist
mkdir -p data/model/sap_sam_2022/filtered/new/google/flan-t5-small/checkpoint-127200/
mkdir -p data/realworld/

# Download and extract the SAP-SAM dataset from Zenodo
# Not needed since I got proper dataset below not having to parse the CSV files
# curl -L -o sap_sam_2022.zip https://zenodo.org/record/7012043/files/sap_sam_2022.zip?download=1
# unzip sap_sam_2022.zip -d ../../data/raw/
# rm sap_sam_2022.zip  # Optional: Remove the zip file after extraction

# Download and extract Kirans xSemAD - dataset TODO: (for later when wanting to train model ourself)
# https://zenodo.org/records/13739241?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImNhYWI3MGQwLTNkZDgtNDVjOC05ODIwLTM3MzkzYzQwMjQxNCIsImRhdGEiOnt9LCJyYW5kb20iOiJmMTU4NWY2ZjJhNjgyNjU1Y2QxMWU1ZjZlZTY5MDE1NyJ9.KWAQjmtMAZytT02G9uDFLDs0FS8P7AWZkD0lyXRnJd9uzl9uKs82ny_cGhDDSsQu-XRrsoRfSogzaghxX_VLmQ
# curl -L -o  constraintCheckingUsingLL.zip https://zenodo.org/records/13736559/files/checkpoint-42400.zip?download=1
# unzip  constraintCheckingUsingLL.zip -d ../../ # TODO: Rename and fix path
# rm  constraintCheckingUsingLL.zip # Optional: Remove the zip file after extraction

# Download and extract Kirans most trained xSemAD model 
# https://zenodo.org/records/13736559?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImYzZGZkNGUyLWIyMjctNGEzNy1hODMzLWM4YTVlYTMwZTA2ZSIsImRhdGEiOnt9LCJyYW5kb20iOiJhMDE0NDE1MTZiMTc3OWZiM2YwMzRmNmUyODkzZjgyZCJ9.BQ0_d_T62NAggiaIGJfGEeWGWN_aJt-AcbHfWDTEYO3p1Wy0hzMlF3ZcWVClT4Ih9vtNjsBdmua4JRU1Im0teg
curl -L -o checkpoint-42400.zip https://zenodo.org/records/13736559/files/checkpoint-42400.zip?download=1
unzip checkpoint-42400.zip -d ../../data/model/sap_sam_2022/filtered/new/google/flan-t5-small/checkpoint-127200/
rm checkpoint-42400.zip # Optional: Remove the zip file after extraction

# Download and extract the XES real-world log
curl -L -o InternationalDeclarations.xes.gz https://data.4tu.nl/file/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/d45ee7dc-952c-4885-b950-4579a91ef426
gunzip InternationalDeclarations.xes.gz
mv InternationalDeclarations.xes ../../data/realworld/

echo "All repositories have been cloned, dataset downloaded, and set up."