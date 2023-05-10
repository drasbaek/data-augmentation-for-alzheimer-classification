#!/usr/bin/env bash
# create virtual environment called aug_alzheimers_env
python3 -m venv aug_alzheimers_env

# activate virtual environment
source ./aug_alzheimers_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt