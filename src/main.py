# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.efficientnet_v2 import (preprocess_input,
                                                 EfficientNetV2M)
# layers
from tensorflow.keras.layers import (Rescaling, Conv2D, MaxPooling2D, Dropout, Dense, Flatten)
# generic model object
from tensorflow.keras.models import Sequential, Model

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

import argparse
import json

def arg_parse():
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()

    # add argument
    parser.add_argument("-a", "--augmentation", type=str, default=None,)

    # parse arguments
    args = parser.parse_args()

    # check if augmentation type is given
    if args.augmentation:
        augmentation = json.loads(args.augmentation)
    
    else:
        augmentation = None

    return augmentation

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
    '''
    Class for creating a custom data generator that allows both original and augmented images to be used simultaneously.

    Args:
    -   original_generator (ImageDataGenerator): Generator for original images.
    -   augmented_generator (ImageDataGenerator): Generator for augmented images.

    '''

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


def load_data_subset(inpath, rel_path, augmentations):
    path = inpath / "dataset_split"

    if rel_path == "train":
        generator = ImageDataGenerator(augmentations, preprocessing_function=preprocess_input)
        shuffle = True
    else:
        generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        shuffle = False

    data_subset = generator.flow_from_directory(
        directory=path / rel_path,
        target_size=(128, 128),
        batch_size=32,
        shuffle=shuffle,
        class_mode="categorical"
    )

    return data_subset


def load_data(inpath, augmentations):
    if augmentations != None:
        train_data_augmented = load_data_subset(inpath, "train", augmentations=augmentations)
        train_data_original = load_data_subset(inpath, "train", augmentations=None)
        train_data = CustomDataGenerator(train_data_original, train_data_augmented)
    
    else:
        train_data = load_data_subset(inpath, "train", augmentations=None)

    # load validation data
    val_data = load_data_subset(inpath, "val", augmentations=None)

    # load test data
    test_data = load_data_subset(inpath, "test", augmentations=None)

    return train_data, val_data, test_data


def build_model():
    '''
    Model inspired from https://www.kaggle.com/code/ashishsingh226/brain-mri-image-alzheimer-classifier/notebook, but made own altercations
    '''

    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(128,128, 3)))
    model.add(Conv2D(filters=16,kernel_size=(7,7),padding='same',activation='relu',kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu',kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.20))

    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation="relu",kernel_initializer="he_normal"))
    model.add(Dense(64,"relu"))
    model.add(Dense(4,"softmax"))

    # print model card
    #model.summary()

    # compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    # parse arguments
    augmentation = arg_parse()

    print(augmentation)

    # define paths
    inpath, outpath = define_paths()

    # split folders
    split_folders(inpath)

    # load data
    train_data, val_data, test_data = load_data(inpath, augmentations=augmentation)

    # build model
    model = build_model()

    # fit the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        batch_size=64,
        epochs=20,
        verbose=1
    )

if __name__ == "__main__":
    main()