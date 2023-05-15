#!/usr/bin/env bash
# create virtual environment called aug_alzheimers_env
python3 -m venv aug_alzheimers_env

# activate virtual environment
source ./aug_alzheimers_env/bin/activate

# install requirements
echo "[INFO:] Installing requirements"
python3 -m pip install -r requirements.txt

# run baseline
echo "[INFO:] Running non-augmented baseline"
python3 ./src/classify.py --name "no_augmentation"

# run augmentations
echo "[INFO:] Running augmentation: increased_brightness"
python3 ./src/classify.py --brightness_range 1.5 1.6 --name "increased_brightness"

echo "[INFO:] Running augmentation: shear"
python3 ./src/classify.py --shear_range 40 --name "shear"

echo "[INFO:] Running augmentation: zoomed_in"
python3 ./src/classify.py --zoom_range 0.75 0.8 --name "zoomed_in"

echo "[INFO:] Running augmentation: rotation"
python3 ./src/classify.py --rotation_range 180 --name "rotation"

echo "[INFO:] All runs complete. Find results in ./out/"

# deactivate virtual environment
deactivate
