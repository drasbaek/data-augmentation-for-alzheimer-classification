from pathlib import Path
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment_image(augmentation_params):
    """
    Augments an image using a given data generator and augmentation parameters.
    Takes the mean over the specified number of channels.

    :param data_generator: Keras ImageDataGenerator instance
    :param augmentation_params: Dictionary of augmentation parameters
    :param num_channels: Number of channels to use for mean calculation
    :return: Augmented image with mean taken over the specified channels
    """
    datagen = ImageDataGenerator(**augmentation_params)
    data_generator = datagen.flow_from_directory(data_generator.directory,
                                                  target_size=data_generator.target_size,
                                                  batch_size=data_generator.batch_size,
                                                  shuffle=data_generator.shuffle,
                                                  class_mode=data_generator.class_mode)
    
    img = data_generator.next()[0][0]
    img = img.mean(axis=2)
    
    return img

# define path
path = Path(__file__)

# define inpath
inpath = path.parents[1] / "in" / "dataset_split" / "train"

# define outpath
outpath = path.parents[1] / "out"

# augment image using data generator (horizontal flip)
augmentation_params = {'shear_range': 30}
horizontal_img = augment_image(augmentation_params)


'''
# augment image using data generator (horizontal flip)
datagen = ImageDataGenerator(shear_range=30)
data_generator = datagen.flow_from_directory(inpath, target_size=(128, 128), batch_size=1, shuffle=False, class_mode="categorical")

# get first image from generator
horizontal_img = data_generator.next()[0][0]

# take mean over all channels
horizontal_img = horizontal_img.mean(axis=2)

# augment image using data generator (increased brightness)
datagen = ImageDataGenerator()
data_generator = datagen.flow_from_directory(inpath, target_size=(128, 128), batch_size=1, shuffle=False, class_mode="categorical")

# get first image from generator
bright_img = data_generator.next()[0][0]

# take mean over all channels
bright_img = bright_img.mean(axis=2)
'''





# plot images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(bright_img)
ax[0].set_title("Original Image")
ax[1].imshow(horizontal_img)
ax[1].set_title("ZCA Whitening")
plt.show()

# save figure with both images
fig.savefig(outpath / "augmented_image.png", dpi=300, bbox_inches="tight")





