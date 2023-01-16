# Model Card

## Model Details
A-Giordano Smith created the model. It is a random forest classifier with n_estimators=50 trained in scikit-learn 1.2.0

## Intended Use
This model should be used to predict a person income on a handful of attributes.

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 

The original data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
The model was evaluated using precision, recall and fbeta.

## Ethical Considerations
The model is trained over socioeeconomics data from 1994, reflecting biases of that age.

## Caveats and Recommendations
This model is a PoC require hyperparameter optimization as well as most recent data to be used.