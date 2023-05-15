# Assignment 4 (Self-Assigned): Data Augmentation for Alzheimers Classification

## Description
This repository forms the solution to self-chosen assignment 4 by Anton Drasbæk Schiønning (202008161) in the course "Visual Analytics" at Aarhus University.

This project attempts to classify varying degrees of the alzheimers diseased based on MRI data using a convolutional neural network. The scope of the analysis is to test whether data augmentation can be used to improve the performance of the model, and if so, which augmentation methods are the most effective?

## Methodology
### Model Architecture
The same architecture is used for all augmentations, a simple convolutional network. In brief, it rescales and uses a convlutional layer to flatten the input images. This flattened image then forms an embedding which is fed to a fully-connected network with three layers. The model card is as follows:
```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
rescaling (Rescaling)       (None, 128, 128, 3)       0

conv2d (Conv2D)             (None, 126, 126, 1)       28

dropout (Dropout)           (None, 126, 126, 1)       0

flatten (Flatten)           (None, 15876)             0

dense (Dense)               (None, 128)               2032256

dense_1 (Dense)             (None, 64)                8256

dense_2 (Dense)             (None, 4)                 260
=================================================================
Total params: 2,040,800
Trainable params: 2,040,800
Non-trainable params: 0
_________________________________________________________________
```

### Selected Augmentations

The explored augmentations on the images are:
* Increased Brightness (Shifts pixel intensities towards higher values)
* Shearing (Rotates the image around its center while stretching it)
* Increased Zoom (Zooms closer into the scan, removing much of the black background)
* Rotation (Rotates image around its center)

![alt text](https://github.com/drasbaek/data-augmentation-for-alzheimer-classification/blob/main/out/aug_illustration.png?raw=True)


## Repository Tree <a name="tree"></a>
```

```

## General Usage <a name="analysis"></a>
### Setup

To run the analysis, you must have Python 3 installed and clone this GitHub repository. <br>
You should also download the [Alzheimers MRI dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) from Kaggle and insert the folder into the `in` directory under the name "Dataset" (should be the default when downloaded)

The analysis is conducted by running the `main.py` file with varying arguments to fulfill the various augmentation types. <br> To run the entire analysis with all augmentations, you should use the `run.sh` bash script as such from the root directory.

### Run

Setting up an environment, installing requirements and running the analysis is all achieved with the following command 
```
bash run.sh
```
This will produce classification reports and loss plots for all augmentation types as well as the non-augmentation baseline in the `out` directory.


## Modified Usage <a name="modified_analysis"></a>
### Setup
To run a modified analysis or only part of the full analysis, you should first install requirements and setup the virtual environment

```
bash setup.sh
```

### Running main.py
The `main.py` script only supports the four specified augmentation types, but their ranges can be altered and the augmentations can be combined. <br> 
To do this, you can run `main.py` directly with the following arguments
```
--brightness_range (default: 1 1) (range used in augmented run: 1.5 1.6)
--shear_range (default: 0) (frange used in augmented run: 40)
--zoom_range (default: 1, 1) (range used in augmented run: 0.75 0.8)
--rotation_range (default: 0) (range used in augmented run: 180)
--name (no default, must be specified)
```
For further information on model parameters, refer to [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

For instance:
```
python3 src/main.py --shear_range 70 --zoom_range 0.4 0.5 --name "shear_and_zoom"
```


## Results
<img width="601" alt="Screenshot 2023-05-14 at 19 42 48" src="https://github.com/drasbaek/data-augmentation-for-alzheimer-classification/assets/80207895/f9849d9b-f9f1-4bc1-bb61-4216170ee1b4">


## Discussion
The results indicate the effectiveness of different data augmentation techniques for detecting Alzheimer's disease. The most impactful augmentation method was increasing brightness in the images, which resulted in an F1-Score of 0.98. This F1-Score was four percentage points higher than the baseline without any augmentation. This finding aligns with previous research that identified increased brightness as the most effective data augmentation type for 3D tumor detection (https://ieeexplore-ieee-org.ez.statsbiblioteket.dk/stamp/stamp.jsp?tp=&arnumber=9506328). <br>

Additionally, increased zoom and rotation also proved to be beneficial augmentations, achieving F1-Scores of 0.97 and 0.96, respectively. <br>

However, the shearing augmentation significantly worsened Alzheimer's detection, resulting in F1-Scores below the baseline. Despite this drawback, shearing was the only augmentation technique that improved the recall for the Moderate Demented class, which was the least represented class in the dataset. Thus, while shearing may not perform well in detecting milder forms of Alzheimer's, it could be considered valuable for accurately identifying moderate Alzheimer's cases.

Still a few limitations to this project should be mentioned:
* It only tests on a narrow scope of images with extremely unbalanced classes.
* It only applies the augmentation techniques in isolation; other effects may appear from using them in conjunction (possible to investigate with the modified usage section)
* The baseline augmentation is only trained on half as much data as the augmentation models and hence should be expected to perform worse under all circumstances.
* Other model structures, such as applying a pretrained CNN instead, could lead to vastly different results for the augmentation types.



