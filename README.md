# Predict-Mechanical-Properties-Low-Alloy-Steel-Using-Structural-Components

## Problem description

The objective of this project is to build multi-target regression ML-model, which allows to predict mechanical properties of low-alloy steel with a given temperature and percentages of structural elements.

This project considers the datasset available in "MatNavi Mechanical properties of low-alloy steels" dataset: https://www.kaggle.com/datasets/konghuanqing/matnavi-mechanical-properties-of-lowalloy-steels?resource=download . 

The features included in the dataset are:

- Alloy code: Identification code
- C:  Carbon (%)
- Si: Silicon (%)
- Mn: Manganese (%)
- P:  Phosphorus (%)
- S:  Sulfur (%)
- Ni: Nickel (%)
- Cr: Chromium (%)
- Mo: Molybdenum (%)
- Cu: Copper (%)
- V:  Vanadium (%)
- Al: Aluminum (%)
- N:  Nitrogen (%)
- Ceq: Carbon Equivalent (%)
- Nb + Ta: Niobium + Titanium (%)
- Temperature (C°)

The target variables are proof stress, tensile strength, elongation and reduction in area.


## EDA

We conducted an Exploratory Data Analysis (EDA) to gain insights into our dataset. 

- We analyze distribution of target variables.

We detect and eliminate an oulier in tensile strength.

- We analyze distribution of features.

We eliminate following features:

Allow code because isn't a structural component.

Ceq because is a combination of other features.

Nb + Ta because has 97.6% of a single value.

- We created a correlation matrix to analyze the relationships between numerical variables.

Since the correlation values between any two variables is not greater than 90%, none of the attributes were removed.


## Model Training

We trained following models:

- Random Forest Regressor
- XGBoost Regressor
- Neural Network

We used k-fold cross-validation to assess model performance.

As a metric we used MAPE (Mean Absolute Percentage Error).


## Tuning Models

For Random Forest hyperparameters tuned included the number of estimators, maximum depth and min_samples_leaf.
 
For XGBoost hyperparameters tuned included number of estimators, maximum depth and learning rate (eta).

For Neural Network hyperparameters tuned included learning rate, drop out rate and size of the inner layer.

We trained and evaluated the models on the validation dataset.


## Select Best Model

To choose the final model, we will train the tuned models with data from full train and evaluate their performance with test

We selected the model with the lowest MAPE score, which is XGBoost.


## Exporting notebook to script

The logic for training the best model is exported to a separate script named `train.py`.


## Virtual Environment

Virtual environment of the project is provided by files `Pipfile` and `Pipfile.lock`. These files contain all information about libraries and dependencies for the project.

To create a virtual environment with libraries and dependencies required for the project, one should first install `pipenv` library:  
   
```
pip install pipenv
```

- Clone the project repository from GitHub.

```
git clone https://github.com/JCGutierrezConcha/Predict-Mechanical-Properties-Low-Alloy-Steel-Using-Structural-Components.git
```

### Installation of virtual environment

Run the following command to install virtual environment:   
   
```
pipenv install
```  

Now you can run `train.py` file with a command

```
pipenv run python train.py
```

Files `predict.py` and `predict_test.py` may be run in a similar way.

### Deploy web service

To deploy web service run the command:

```
pipenv run waitress-serve --listen=localhost:9696 predict:app
```

### Usage app to predict

Open a new terminal, be sure requests is installed. If is necessary use ```pip install requests```.

Then run  ```python predict_test.py```.

## Run with Docker

Once you have cloned the repo, you need to have Docker installed on your machine and just build and run the docker image.

To build the image run
```
docker build -t {build-tag} .
```
`{build-tag}`: Specifies any user-defined tag for docker image. eg. `app-test`


To run the image run
```
docker run -it --rm -p 9696:9696 {build-tag}
```

- -it enables an interactive terminal.
- --rm removes the container when it exits.
- -p 9696:9696 maps port 9696 on your host to port 9696 within the container.


### Usage app to predict

Open a new terminal, be sure requests is installed. If is necessary use ```pip install requests```.

Then run  ```python predict_test.py```.


## Sample Output

![Sample of the project running locally](https://github.com/JCGutierrezConcha/Predict-Mechanical-Properties-Low-Alloy-Steel-Using-Structural-Components/blob/main/deploy_predict_local.PNG)


