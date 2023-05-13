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
python3 ./src/main.py --name "no_augmentation"

# run augmentations
echo "Running augmentation: increased_brightness"
python3 ./src/main.py --brightness_range 1.5 1.6 --name "increased_brightness"

echo "Running augmentation: shear"
python3 ./src/main.py --shear_range 40 --name "shear"

echo "Running augmentation: zoomed_in"
python3 ./src/main.py --zoom_range 0.75 0.8 --name "zoomed_in"

echo "Running augmentation: rotation"
python3 ./src/main.py --rotation_range 180 --name "rotation"
