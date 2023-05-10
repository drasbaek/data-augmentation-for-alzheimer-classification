# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.metrics import classification_report

# for plotting
import matplotlib.pyplot as plt

from pathlib import Path

# splitting folders
import splitfolders

import pandas as pd
import numpy as np
from keras.utils import Sequence


def define_paths():
    '''
    Define paths for input and output data.
    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   outpath (pathlib.PosixPath): Path to where the classified data should be saved.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "in"

    # define output dir
    outpath = path.parents[1] / "out"

    return inpath, outpath

def split_folders(inpath):
    '''
    Split the dataset into train, validation and test folders.
    These folders are saved in the same directory as the input data.
    Args:
    -   inpath (pathlib.PosixPath): Path to input data.
    '''

    # define output dir
    outpath = inpath / "dataset_split"

    # check if the folder exists already
    if not outpath.exists():
        # create the folder
        outpath.mkdir()
        
        # split the data into train, validation and test folders
        splitfolders.ratio(inpath, output=outpath, seed=2502, ratio=(.75, .125, .125))
    

class CustomDataGenerator(Sequence):
    def __init__(self, original_generator, augmented_generator):
        self.original_generator = original_generator
        self.augmented_generator = augmented_generator
        
    def __len__(self):
        return len(self.original_generator)
    
    def __getitem__(self, index):
        # get the original batch of images and labels
        x_original, y_original = self.original_generator[index]
        
        # get the augmented batch of images and labels
        x_augmented, y_augmented = self.augmented_generator[index]
        
        # concatenate the two batches
        x = np.concatenate([x_original, x_augmented], axis=0)
        y = np.concatenate([y_original, y_augmented], axis=0)
        
        return x, y


def load_data_subset(inpath, rel_path, shuffle=True, augmentations=None):
    path = inpath / "dataset_split"

    if rel_path == "train":
        generator = ImageDataGenerator(augmentations)
    else:
        generator = ImageDataGenerator()

    data_subset = generator.flow_from_directory(
        directory=path / rel_path,
        target_size=(128, 128),
        batch_size=32,
        shuffle=shuffle,
        class_mode="categorical"
    )

    return data_subset


def load_data(inpath, rel_path, shuffle=True, augmentations=None):
    if augmentations != None:
        train_data_augmented = load_data_subset(inpath, "train", shuffle=True, augmentations=augmentations)
        train_data_original = load_data_subset(inpath, "train", shuffle=True)
        train_data = CustomDataGenerator(train_data_original, train_data_augmented)
    
    else:
        train_data = load_data_subset(inpath, "train", shuffle=True)

    # load validation data
    val_data = load_data_subset(inpath, "val", shuffle=False)

    # load test data
    test_data = load_data_subset(inpath, "test", shuffle=False)

    return train_data, val_data, test_data


def main():
    # define paths
    inpath, outpath = define_paths()

    # split folders
    split_folders(inpath)

    # define augmentations
    augmentations = None

    # load data
    train_data, val_data, test_data = load_data(inpath, "train", shuffle=True, augmentations=augmentations)




if __name__ == "__main__":
    main()