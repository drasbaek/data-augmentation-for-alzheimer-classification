# Assignment 4 (Self-Assigned): Data Augmentation for Alzheimers Classifications

## Description
This repository forms the solution to self-chosen assignment 4 by Anton Drasbæk Schiønning (202008161) in the course "Visual Analytics" at Aarhus University.

This project attempts to classify varying degrees of the alzheimers diseased based on MRI data using a pre-trained CNN. The scope of the analysis is to test whether data augmentation can be used to improve the performance of the model, and if so, which augmentation methods are the most effective?

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
To do this, you can run `main.py` directly, specifying arguments as such
```
```


## Results
<img width="504" alt="Screenshot 2023-05-12 at 08 41 29" src="https://github.com/drasbaek/data-augmentation-for-alzheimer-classification/assets/80207895/9dce8870-0dda-4d0a-ac5a-d5e8112f314b">


