#!/usr/bin/env bash
# create virtual environment called aug_alzheimers_env
python3 -m venv aug_alzheimers_env

# activate virtual environment
source ./aug_alzheimers_env/bin/activate

# install requirements
echo "Install Requirements"
python3 -m pip install -r requirements.txt

# run baseline
echo "Running non-augmented baseline"
python3 ./src/main.py

# run augmentations
echo "Running augmentation: increased_brightness"
python3 ./src/main.py --brightness_range 1.2 1.7 --name "increased_brightness"

echo "Running augmentation: zca_whitening"
python3 ./src/main.py --zca_whitening True --name "zca_whitening"

echo "Running augmentation: zoomed_in"
python3 ./src/main.py --zoom_range 1.2 1.7 --name "zoomed_in"

echo "Running augmentation: zoomed_out"
python3 ./src/main.py --horizontal_flip True --name "horizontal_flip"
