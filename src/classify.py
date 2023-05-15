""" classifier.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script contains the code for a CNN classifier that identifies varying degrees of alzheimers in brain scans.
    It is designed to be run from the command line, where it is possible to specify augmentation parameters.

Usage:
    $ python src/classifier.py
"""

# import packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Rescaling, Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import splitfolders
import numpy as np
from keras.utils import Sequence
import argparse

# define functions
def arg_parse():
    """
    Parse command line arguments to script.
    It is possible to specify the augmentation parameters to be used and their ranges.
    You must also specify a name for the model output to be saved under

    Returns:
      args (argparse.Namespace): Parsed arguments.
    """

    # initialize parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("-b", "--brightness_range", nargs="+", type=float, default=[1, 1])
    parser.add_argument("-s", "--shear_range", type=int, default=0)
    parser.add_argument("-z", "--zoom_range", nargs="+", type=float, default=[1, 1])
    parser.add_argument("-r", "--rotation_range", type=int, default=0)
    parser.add_argument("-n", "--name", type=str)

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

    # define output dir for split data
    outpath = inpath / "dataset_split"

    # check if the folder exists already (this ensures that we only do the split for the first run)
    if not outpath.exists():
        print("Splitting Data...")
        
        # split the data into train, validation and test folders
        splitfolders.ratio(inpath / "Dataset", output=outpath, seed=2502, ratio=(.7, .1, .2))
    

class CustomDataGenerator(Sequence):
    '''
    Class for creating a custom data generator that allows both original and augmented images to be used simultaneously.
    This is necessary as we cannot simply merge the data as we are using generators.

    Args:
    -   original_generator (ImageDataGenerator): Generator for original images.
    -   augmented_generator (ImageDataGenerator): Generator for augmented images.

    '''

    # initialize the class
    def __init__(self, original_generator, augmented_generator):
        self.original_generator = original_generator
        self.augmented_generator = augmented_generator
    
    # define the length of the generator
    def __len__(self):
        return len(self.original_generator)
    
    # define the getitem method
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
    """
    Loads a subset of the data (train, test or validation) and returns it as a generator.

    Args:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   rel_path (str): Relative path to the subset of the data to be loaded (i.e. train, test or validation).
    -   args (argparse.Namespace): Parsed arguments (containing augmentation parameters).

    Returns:
    -   data_subset (ImageDataGenerator): Generator for the subset of the data.
    """

    # get path to the split dataset
    path = inpath / "dataset_split"
    
    # checks if we use augmentation
    if rel_path == "train" and args != None:
        generator = ImageDataGenerator(brightness_range = args.brightness_range, 
                                        shear_range = args.shear_range,
                                        zoom_range = args.zoom_range,
                                        rotation_range = args.rotation_range)
        shuffle = True
    
    # check if we use training dat, must be shuffled
    elif rel_path == "train" and args == None:
        generator = ImageDataGenerator()
        shuffle = True
    
    # else validation or test data, must not be shuffled
    else:
        generator = ImageDataGenerator()
        shuffle = False

    # load the data
    data_subset = generator.flow_from_directory(
        directory=path / rel_path,
        target_size=(128, 128),
        batch_size=32,
        shuffle=shuffle,
        class_mode="categorical"
    )

    return data_subset


def load_all_data(inpath, args):
    """
    Utilizes load_data_subset recurrently to load all data (train, test and validation) and returns it as a generator.
    Uses CustomDataGenerator to merge original and augmented data if augmentation is used.

    Args:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   args (argparse.Namespace): Parsed arguments (containing augmentation parameters).

    Returns:
    -   train_data (ImageDataGenerator): Generator for the training data.
    -   val_data (ImageDataGenerator): Generator for the validation data.
    -   test_data (ImageDataGenerator): Generator for the test data.
    """
    
    # check if all arguments are default (i.e. no augmentation)
    if all([
        args.brightness_range == [1, 1],
        args.shear_range == 0,
        args.zoom_range == [1, 1],
        args.rotation_range == 0
    ]):
        train_data = load_data_subset(inpath, "train")

    # else load augmented and original data and merge them
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
    Builds the model for the classification task.
    The model is a simple convolutional neural network with a dense layer on top.

    Returns:
    -   model (tensorflow.python.keras.engine.sequential.Sequential): The compiled model.
    '''

    # initialize the model
    model = Sequential()

    # create a one dimensional embedding of the image
    model.add(Rescaling(1./255, input_shape=(128,128,3)))
    model.add(Conv2D(filters=1,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Dropout(0.3)) # big dropout to avoid overfitting
    model.add(Flatten())

    # classify the embedding with a simple dense network
    model.add(Dense(128,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(4,"softmax"))

    # compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def initialize_outputs(outpath, args):
    """
    Initializes an output folder (for clf report and loss/accuracy plots) for the specific run.

    Args:
    -   outpath (pathlib.PosixPath): Path to where the output folder should be created.
    -   args (argparse.Namespace): Namespace object containing the arguments passed to the script.
    """

    # get name for output folder
    name = args.name

    # name a new folder in output after name if it does not exist already
    if not (outpath / name).exists():
        (outpath / name).mkdir()


def plot_history(history, outpath, args):
    """
    Creates and saves a plot displaying the training/validation loss and accuracy.

    Args:
    -   history (tensorflow.python.keras.callbacks.History): History object containing the training and validation loss and accuracy.
    -   outpath (pathlib.PosixPath): Path to where the plots should be saved.
    -   args (argparse.Namespace): Namespace object containing the arguments passed to the script.
    """

    # create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # plot training and validation loss
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_title(f'{args.name}: Training and validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # plot training and validation accuracy
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='val')
    ax2.set_title(f'{args.name}: Training and validation accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # adjust the spacing between subplots
    fig.tight_layout()

    # save the plot
    plt.savefig(outpath / args.name / f"{args.name}_loss_and_accuracy.png")

def save_classification_report(outpath, args, y_true, y_pred, target_names):
    """
    Saves a classification report as a text file.

    Args:
    -   outpath (pathlib.PosixPath): Path to where the classification report should be saved.
    -   args (argparse.Namespace): Namespace object containing the arguments passed to the script.
    -   y_true (numpy.ndarray): Array containing the true labels.
    -   y_pred (numpy.ndarray): Array containing the predicted labels.
    -   target_names (list): List containing the names of the classes (neccessary for the classification to look nice)
    """

    # create classification report
    report = classification_report(y_true, y_pred, target_names=target_names)

    # save report
    with open(outpath / args.name / f"{args.name}_clf_report.txt", "w") as f:
        f.write(report)
    

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

    # fit the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        batch_size=64,
        epochs=10,
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