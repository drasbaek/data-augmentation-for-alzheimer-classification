""" illustration.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
   This script illustrates the augmentations used in the final model.
   It is only used to create an illustration for the project README.md file.

Usage:
    $ python src/illustration.py
"""

# load packages
from pathlib import Path
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def augment_image(inpath, augmentation_params):
    """
    Augment image using ImageDataGenerator.
    It only augments a single image, to be used for visualization.

    Args:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   augmentation_params (dict): Dictionary of augmentation parameters.

    Returns:
    -   img (np.array): Augmented image.
    """

    # define data generator
    datagen = ImageDataGenerator(**augmentation_params)

    # create generator
    data_generator = datagen.flow_from_directory(inpath,
                                                  target_size=(128, 128),
                                                  batch_size=1,
                                                  shuffle=False,
                                                  class_mode="categorical")
    # get image
    img = data_generator.next()[0][0]

    # convert to grayscale (it already is grayscale, but has 3 channels)
    img = img.mean(axis=2)
    
    return img


def illustrate_augmentations(images, titles, outpath):
    """
    Illustrate augmentations on a single plot.

    Args:
    -   images (list): List of images.
    -   titles (list): List of titles.
    -   outpath (pathlib.PosixPath): Path to output directory.
    """

    # set up plot parameters
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    fig.subplots_adjust(wspace=0.05)

    # plot images
    for i in range(len(images)):
        ax[i].imshow(images[i], cmap="gray")
        ax[i].set_title(titles[i], fontsize=12, fontweight="bold", fontfamily="serif")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()

    # save figure with both images
    fig.savefig(outpath / "aug_illustration.png", dpi=600, bbox_inches="tight")


def main():
    # define path
    path = Path(__file__)

    # define inpath
    inpath = path.parents[1] / "in" / "dataset_split" / "train"

    # define outpath
    outpath = path.parents[1] / "out"

    # define all augmentations as a list of dictionaries
    augmentations = [
        {},
        {'brightness_range': [1.5, 1.6]},
        {'shear_range': 40},
        {'zoom_range': [0.75, 0.8]},
        {'rotation_range': 180}
    ]

    # define list of titles
    titles = [
        "No Augmentation",
        "Increased Brightness",
        "Shearing",
        "Increased Zoom",
        "Rotation"]

    # get image using each augmentation
    images = []

    for augmentation in augmentations:
        img = augment_image(inpath, augmentation)
        images.append(img)

    # illustrate augmentations and save figure
    illustrate_augmentations(images, titles, outpath)

if __name__ == "__main__":
    main()