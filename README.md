# SmartWF: Learning-Based Secure Wind Farm Control

This project is part of the TI3165TU Capstone Applied AI Project course as part of the Engineering with AI minor offered at TU Delft. The repository is split up into 3 parts, each contaning different models with different architectures, inptus and complexities. The models aim to predict the loads or damage equivalent loads experienced by turbines in normal operating conditions.

## Models
This repository contains 3 different styles of models, the ```Must```, ```Should``` and ```Could``` model, which increase in complexity and predictive capabilities each using different inputs and giving different outputs.

### Must
The `Must` model is the most simple, it aims to make basic predictions on the Damage Equivalent Load (DEL) of a wind turbine. It uses average wind velocity and the standard deviation of the wind as input.

There are two complete `Must` model notebooks trying to predict Leq_x or Leq_y. (this is Damage Equivalent Load for Root Bending Moment x and y).
Both models use the same traditional AI techniques. A KNN-model, Random Forest and Polynomial SGD.

Both Notebooks use 8 plotfunctions defined in seperate 'Plot_data.py'. These plotfunctions make sure all output graphs have similar layout.

To see our tuning result the following steps should be followed
1. Open the `Traditional AI Techniques_Leq_x(version 1).ipynb` or `Traditional AI Techniques_Leq_y(version 1).ipynb` from the Must\Traditional_Ai_techniques folder. All outputs should be there already. 

To train a model the following steps should be followed:
1. Open the `Traditional AI Techniques_Leq_x(version 1).ipynb` or `Traditional AI Techniques_Leq_y(version 1).ipynb` from the Must\Traditional_Ai_techniques folder
2. Specify the folderpath of the `DEL_must_model_rep_{number}` files.
3. Choose at the end of each model wich plots are desired. Meaning and effect of input variables can be found in `Plot_data.py`.
4. Run all cels.

### Should
The ```Should``` has increased predictive capabilities, being able to predict a time series of the blade root bending moment of the x- and y-axis in the blade reference system.

The inputs are (as a time series):
- The average wind speed in the x-direction
- The azimuth angle of the blad for which loads are predicted

The outputs are (as a time series):
- The loads in direction to the specified axis

To train a model the following steps should be followed:
1. Open the `training_notebook.ipynb` jupyter notebook in the Should folder of the GitHub.
2. Specify the `dataset path`, `save directory` and desired `load_axis` variables in the file.
3. Run all cells.

For inference the following steps should be followed:
1. Open the `inference_notebook.ipynb` jupyter notebook in the Should folder of the GitHub.
2. Specify the `dataset_path`, `model_path`, `load_axis` and `test_files` variables in the file.
3. Run all cells.

### Could

## Dataset
The `Must` dataset was created using the openFAST simulator with the Servodyn, Aerodyn and Elastodyn modules installed. Dataset was generated for wind speeds of 5 m/s to 25 m/s. To simulate turbulent wind a standard deviation of 0.25 m/s to 2.50 m/s was used. The windflow is assumed to be uniform through space while varying through time, this means that turbulent effects or wind shear are not considered.  
These inputfiles are used for the openFast simulation software and output variables Timestamp, RootMy1b en RootMx1b are stored. These are the bending moments at the root of balde 1. From these bendingmoments a Damage Equivalent Load (or Load Equivalent, Leq) is calculated using Rainflow counting algorithm.

The `Should` dataset was created using the openFAST simulator with the Servodyn, Aerodyn and Elastodyn modules installed. Dataset was generated for wind speeds of 5 m/s to 25 m/s. To simulate turbulent wind a standard deviation of 0.25 m/s to 2.50 m/s was used. The windflow is assumed to be uniform through space while varying through time, this means that turbulent effects or wind shear are not considered.

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
