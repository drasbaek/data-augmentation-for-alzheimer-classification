# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# layers
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# generic model object
from tensorflow.keras.models import Sequential

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
    parser.add_argument("-b", "--brightness_range", nargs="+", type=float, default=[1, 1])
    parser.add_argument("-w", "--zca_whitening", type=bool, default=False)
    parser.add_argument("-z", "--zoom_range", nargs="+", type=float, default=[1, 1])
    parser.add_argument("-f", "--horizontal_flip", type=bool, default=False)
    parser.add_argument("-n", "--name", type=str, default="no_augmentation")


    # parse arguments
    args = parser.parse_args()

    return args


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

    # check if the folder exists already (this ensures that we only do the split for the first run)
    if not outpath.exists():
        print("Splitting Data...")
        
        # split the data into train, validation and test folders
        splitfolders.ratio(inpath / "Dataset", output=outpath, seed=2502, ratio=(.7, .1, .2))
    

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


def load_data_subset(inpath, rel_path, args=None):
    # get path to the split dataset
    path = inpath / "dataset_split"

    if rel_path == "train" and args != None:
        generator = ImageDataGenerator(brightness_range = args.brightness_range, 
                                        zca_whitening = args.zca_whitening,
                                        zoom_range = args.zoom_range,
                                        horizontal_flip = args.horizontal_flip)
        shuffle = True
    
    elif rel_path == "train" and args == None:
        generator = ImageDataGenerator()
        shuffle = True
    else:
        generator = ImageDataGenerator()
        shuffle = False

    data_subset = generator.flow_from_directory(
        directory=path / rel_path,
        target_size=(128, 128),
        batch_size=32,
        shuffle=shuffle,
        class_mode="categorical"
    )

    return data_subset


def load_all_data(inpath, args):
    # check if all arguments are default
    if all([
        args.brightness_range == [1, 1],
        not args.zca_whitening,
        args.zoom_range == [1, 1],
        not args.horizontal_flip
    ]):
        train_data = load_data_subset(inpath, "train")
    
    else:
        train_data_augmented = load_data_subset(inpath, "train", args)
        train_data_original = load_data_subset(inpath, "train")
        train_data = CustomDataGenerator(train_data_original, train_data_augmented)

    # load validation data
    val_data = load_data_subset(inpath, "val")

    # load test data
    test_data = load_data_subset(inpath, "test")

    return train_data, val_data, test_data


def build_model():
    '''
    Model inspired from https://www.kaggle.com/code/ashishsingh226/brain-mri-image-alzheimer-classifier/notebook, but heavily simplified and small alterations made.
    '''
    model = Sequential()

    # create a 1-dimensional embedding
    model.add(Rescaling(1./255, input_shape=(128,128,3)))
    model.add(Conv2D(filters=1,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.4)) # big dropout to avoid overfitting
    model.add(Flatten())

    # classify with 3 layer NN
    model.add(Dense(128,activation="relu"))
    model.add(Dense(64,activation="relu"))
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

def initialize_outputs(outpath, args):
    # get name for output folder
    name = args.name

    # name a new folder in output after name if it does not exist already
    if not (outpath / name).exists():
        (outpath / name).mkdir()

def plot_history(history, outpath, args):
    '''
    Creates and saves a plot displaying the training and validation loss and accuracy.

    Args:
    -   history (tensorflow.python.keras.callbacks.History): History object containing the training and validation loss and accuracy.
    -   outpath (pathlib.PosixPath): Path to where the plots should be saved.
    '''

    # Create a new figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Plot training and validation loss
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_title(f'{args.name}: Training and validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation accuracy
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='val')
    ax2.set_title(f'{args.name}: Training and validation accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot
    plt.savefig(outpath / args.name / f"{args.name}_loss_and_accuracy.png")

def save_classification_report(outpath, args, y_true, y_pred, target_names):
    '''
    Saves a classification report as a text file.

    Args:
    -   outpath (pathlib.PosixPath): Path to where the classification report should be saved.
    -   y_true (numpy.ndarray): Array containing the true labels.
    -   y_pred (numpy.ndarray): Array containing the predicted labels.
    -   target_names (list): List containing the names of the classes (neccessary for the classification to look nice)

    '''

    # create classification report
    report = classification_report(y_true, y_pred, target_names=target_names)

    # save report
    
    with open(outpath / args.name / f"{args.name}_clf_report.txt", "w") as f:
        f.write(report)
    
    print(report)
    

def main():
    # parse arguments
    args = arg_parse()

    # define paths
    inpath, outpath = define_paths()

    # split folders
    split_folders(inpath)

    # load data
    train_data, val_data, test_data = load_all_data(inpath, args)

    # build model
    model = build_model()

    model.summary()

    # fit the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        batch_size=16,
        epochs=15,
        verbose=1
    )

    # initialize output folder
    initialize_outputs(outpath, args)

    # plot history
    plot_history(history, outpath, args)

    # get predicted classes
    y_pred = model.predict(test_data).argmax(axis=1)

    # save classification report
    save_classification_report(outpath, args, test_data.classes, y_pred, test_data.class_indices.keys())



if __name__ == "__main__":
    main()