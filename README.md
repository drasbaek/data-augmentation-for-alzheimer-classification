# Assignment 4 (Self-Assigned): Data Augmentation for Alzheimers Classification

## Description
This repository forms the solution to self-chosen assignment 4 by Anton Drasbæk Schiønning (202008161) in the course "Visual Analytics" at Aarhus University.

This project attempts to classify varying degrees of the alzheimers diseased based on MRI data using a convolutional neural network. The scope of the analysis is to test whether data augmentation can be used to improve the performance of the model, and if so, which augmentation methods are the most effective?

## Methodology
The model used for classifying the images is a simple convolutional network. For the full model stucture, please refer to the function `build_model()` in `main.py`. The model structure is the same for all augmentations. <br>

The explored augmentations on the images are:
* Increased Brightness
* Shearing
* Increased Zoom
* Rotation

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
<img width="504" alt="Screenshot 2023-05-12 at 08 41 29" src="https://github.com/drasbaek/data-augmentation-for-alzheimer-classification/assets/80207895/9dce8870-0dda-4d0a-ac5a-d5e8112f314b">

## Discussion
The results indicate that data augmentation proved useful in detecting alzheimers overall. The most useful method proved to be increasing brightness in the images as this gave an F1-Score of xx, which is xx over the no augmentation baseline. This conforms with findings for most effective data augmentation type for 3D tumor detection (https://ieeexplore-ieee-org.ez.statsbiblioteket.dk/stamp/stamp.jsp?tp=&arnumber=9506328). Also, xx and xx proved to be useful augmentations



