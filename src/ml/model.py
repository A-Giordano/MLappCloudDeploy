from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=50)
    clf.fit(X_train,y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForest model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def to_pickle(obj, name: str):
        with open(f"starter/model/{name}.pkl", "wb") as file:
            pickle.dump(obj, file)

def save_performance_on_slices(X_raw, preds, y_test, categorical_features=[]):
    """ Function for calculating model performances on slices of the dataset."""
    # preds = model.predict(X_test)
    X_raw = X_raw.reset_index(drop=True)
    output_file_lines = []
    with open('starter/model/slice_output.txt', 'w') as f:
        for cat in categorical_features:
            f.write("--------------------------\n")
            f.write(f"Performance on category: {cat}\n")
            output_file_lines.append
            for cls in X_raw[cat].unique():
                bln_mask = X_raw[cat] == cls
                tmp_preds = preds[bln_mask]
                tmp_y = y_test[bln_mask]
                fbeta = fbeta_score(tmp_y,tmp_preds, beta=1, zero_division=1)
                precision = precision_score(tmp_y, tmp_preds, zero_division=1)
                recall = recall_score(tmp_y, tmp_preds, zero_division=1)
                f.write(f"[precision: {precision}, recall: {recall}, fbeta: {fbeta}],Class: {cls}, samples: {tmp_preds.shape[0]}\n")
            f.write("--------------------------\n\n")

