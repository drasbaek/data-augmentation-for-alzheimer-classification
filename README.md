# Assignment 4 (Self-Assigned): Data Augmentation for Alzheimer Classification

## Repository Overview
1. [Description](#description)
2. [Repository Tree](#tree)
3. [Methodology](#method)
5. [General Usage](#gusage)
6. [Modified Usage](#musage)
7. [Results](#result)
8. [Discussion](#discuss)

## Description <a name="description"></a>
This repository forms the solution to self-chosen assignment 4 by Anton Drasbæk Schiønning (202008161) in the course "Visual Analytics" at Aarhus University.

This project attempts to classify varying degrees of the Alzheimer's Disease based on MRI data using a convolutional neural network. Specifically, it investigates whether data augmentation techniques can improve performance of the network. The motivation behind this is using a data-driven approach to better identify a disease which hampers people's relationships, memories and personal quality of life across cultures.

The scope of the analysis is to test:
1. Can data augmentation can be used to improve Alzheimer's detection?
2. If so, which augmentation methods are the most effective in Alzheimer's detection?
<br>

## Repository Tree <a name="tree"></a>
```
├── in
│   └── Dataset             <----- Alzheimers MRI dataset (MUST BE DOWNLOADED AND INSERTED HERE)
├── out
│   ├── aug_illustration.png                          
│   ├── increased_brightness                          
│   │   ├── increased_brightness_clf_report.txt         <----- Classification report with scores for increased brightness
│   │   └── increased_brightness_loss_and_accuracy.png  <----- Loss/Accuracy for training and validation with increased brightness
│   ├── no_augmentation
│   ├── rotation
│   ├── shear
│   └── zoomed_in
├── requirements.txt
├── run.sh
├── setup.sh
└── src
    ├── illustration.py     <----- Script for creating illustration of augmentations  
    └── classify.py         <----- Script for running a classification model on an augmentation type
```
<br/><br/>

## Methodology <a name="method"></a>
### Model Architecture
A fairly simple convolutional neural network, build from scratch, is used for all runs. In brief, it rescales and uses a convlutional layer to flatten the input images. This flattened image then forms an embedding which is fed to a fully-connected network with three layers. The model card is as follows:
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
For each augmentation type, a model of this architecture was fit to the data and trained for a duration of 10 epochs. <br/><br/>

### Selected Augmentations

The explored augmentations on the images are:
* Increased Brightness (Shifts pixel intensities towards higher values)
* Shearing (Rotates the image around its center while stretching it)
* Increased Zoom (Zooms closer into the scan, removing much of the black background)
* Rotation (Rotates image around its center)

![alt text](https://github.com/drasbaek/data-augmentation-for-alzheimer-classification/blob/main/out/aug_illustration.png?raw=True)
<br/><br/>

## General Usage <a name="gusage"></a>
### Setup

To run the analysis, you must have Python 3 installed and clone this GitHub repository. You should also download the [Alzheimers MRI dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) from Kaggle and insert the folder into the `in` directory under the name "Dataset" (should be the default when downloaded) as seen on the *Repository tree*.

### Run

The analysis is conducted by running the `classify.py` file with varying arguments to fulfill the various augmentation types. To run the entire analysis with all augmentations, you should use the `run.sh` bash script as such from the root directory.
```
bash run.sh
```

This achieves the following:
* Sets up a virtual environment
* Installs requirements to that environment
* Runs classification using the dataset with no augmentations
* Runs classifications for each of the augmentation types
* Deactivates environment

This will produce classification reports and loss/accuracy plots for all augmentation types as well as the no augmentation baseline in the `out` directory.
<br/><br/>

## Modified Usage <a name="musage"></a>
### Setup
To run a modified analysis or only part of the full analysis, you should follow the setup steps from *General Usage*. In addition, a shell script must be run to activate virtual environment and install requirements

```
bash setup.sh
```

### Running the analysis
The `main.py` script only supports the four specified augmentation types, but their ranges can be altered and the augmentations can be combined. To do this, you can run `main.py` directly with the following arguments:

| Argument              | Default | Value(s) used in Augmented Run                                                             |
|---------------------|---------------|-------------------------------------------------------------------------|
| `--brightness_range`       | 1.0 1.0        | 1.5 1.6       |
| `--shear_range`      | 0           | 40                                          |
| `--zoom_range`       | 1.0 1.0           | 0.75 0.8                                          |
| `--rotation_range`       | 0           | 180                                          |
| `--name`       | *None, must be specified*           |        *Varies by run*   
<br>

For instance:
```
python3 src/main.py --shear_range 70 --zoom_range 0.4 0.5 --name "shear_and_zoom"
```
The folder in `out` is named according to what you specify under the `name` argument. For further information on model parameters, refer to [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).
<br/><br/>

## Results <a name="result"></a>
<img width="624" alt="Screenshot 2023-05-23 at 14 44 33" src="https://github.com/drasbaek/data-augmentation-for-alzheimer-classification/assets/80207895/b7c848e2-563d-4c9a-874d-c7fcba7d9dcc">

<br/><br/>

## Discussion <a name="discuss"></a>
The results indicate the effectiveness of different data augmentation techniques for detecting Alzheimer's disease. The most impactful augmentation method was increasing brightness in the images, which resulted in an **F1-Score of 0.98**. This F1-Score was four percentage points higher than the baseline without any augmentation. This finding aligns with previous research that identified increased brightness as the most effective data augmentation type for 3D tumor detection (Cirillo et al., 2021). <br>

Additionally, increased zoom and rotation also proved to be beneficial augmentations, achieving F1-Scores of 0.97 and 0.96, respectively. <br>

However, the shearing augmentation significantly worsened Alzheimer's detection, resulting in F1-Scores below the baseline. Despite this drawback, shearing was the only augmentation technique that improved the recall for the *Moderate Demented* class, which was the least represented class in the dataset. Thus, while shearing may not perform well in detecting milder forms of Alzheimer's, it could be considered valuable for accurately identifying moderate Alzheimer's cases.

Still a few limitations to this project should be mentioned:
* It only tests on a narrow scope of images with extremely unbalanced classes.
* It only applies the augmentation techniques in isolation; other effects may appear from using them in conjunction (possible to investigate with the modified usage section)
* The baseline augmentation is only trained on half as much data as the augmentation models and hence should be expected to perform worse under all circumstances.
* Other model structures, such as applying a pretrained CNN instead, could lead to vastly different results for the augmentation types.

## References
* Cirillo, M. D., Abramian, D., & Eklund, A. (2021, September). What is the best data augmentation for 3D brain tumor segmentation?. In 2021 IEEE International Conference on Image Processing (ICIP) (pp. 36-40). IEEE.

