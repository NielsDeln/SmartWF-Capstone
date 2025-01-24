# SmartWF: Learning-Based Secure Wind Farm Control

This project is part of the TI3165TU Capstone Applied AI Project course as part of the Engineering with AI minor offered at TU Delft. The repository is split up into 3 parts, each contaning different models with different specifications and

## Models
This repository contains 3 different styles of models, the ```Must```, ```Should``` and ```Could``` model. These models increase in complexity and capabilities 

### Must
The ```Must``` model is the most simple, it aims to make basic predictions on the Damage Equivalent Load (DEL) of a wind turbine based on the wind velocity and the standard deviation of the wind.

There are two complete ```Must``` model notebooks. One trying to predict Leq_x and Leq_y. (is Damage Equivalent Load for Root Bending Moment x and y).
Both models use the same traditional AI techniques. A KNN-model, Random Forest and Polynomial SGD.

Both Notebooks use 8 plotfunctions defined in seperate Plot_data.py. These plotfunctions make sure all output graphs have similar layout.

### Should
The ```Should``` has increased predictive capabilities, being able to predict a time series of the blade root bending moment 

### Could

## Dataset
The `Must` and `Should` models are trained using the same dataset. This dataset was created using the openFAST simulator with the Subdyn, Aerodyn and Elastodyn modules installed. Dataset was generated for wind speeds of 5 m/s to 25 m/s. To simulate turbulent wind a standard deviation of 0.25 m/s to 2.50 m/s was used. The complete raw dataset can be made available upon request at n.delnoij@studnet.tudelft.nl , due to the size of the dataset

## Installation
The GitHub repo can be cloned to your local envoirnment for use of the functions or any of the supplied trained models with the following command:

```shell
  git clone https://github.com/NielsDeln/SmartWF-Capstone
```
### Required Packages
All packages required for training and performing inference of the model in this repo are stated in ```requirements.txt```. They can be installed as follows:
```shell
  pip install -r requirements.txt
```

This requires that the command is exectued in the same directory as the location of `requirements.txt`

The openFAST toolbox is only required for editing the dataset and for the calcualtion of damage equivalent loads for the `Must` model. This has already been done for the existing dataset.

The openFAST toolbox can it's installation procedure can be found on it's GitHub repo `https://github.com/OpenFAST/openfast_toolbox`.