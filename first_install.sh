#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

cd constraints-transformer/

# Install BLEURT from GitHub
pip3 install git+https://github.com/google-research/bleurt.git

# Install transformer4bpm
pip3 install git+https://github.com/disola/bpart5.git

pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz --no-deps

# Install the rest of the requirements
pip3 install -r requirements.txt

