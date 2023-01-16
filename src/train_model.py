# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_performance_on_slices, to_pickle
import pandas as pd

# Load in the data.
data = pd.read_csv("data/census.csv")
print(data.shape)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)
print('train', train.shape)
print('test', test.shape)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
# Train and save a model.
model = train_model(X_train, y_train)
to_pickle(model, "inference_model")
to_pickle(encoder, "encoder")
to_pickle(lb, "label_binarizer")

y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
print(precision, recall, fbeta)
save_performance_on_slices(data.iloc[test.index], y_preds, y_test, cat_features)
