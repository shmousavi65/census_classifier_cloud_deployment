import itertools, pickle
import pandas as pd
import mlflow
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer

def get_training_inference_pipeline(categorical_features, numerical_features, model_params):
    """
    return the full input pipeline.

    Inputs
    ------
    categorical_features (list): list of categorical features 
    numerical_features (list): list of numerical features 
    model_params (dict): ml model parameters 
    Returns
    -------
    model (Pipeline): the pipeline including the trasnformers and the ml model
    """
    # We need 2 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    
    # Categorical preprocessing pipeline
    categorical_features = sorted(categorical_features)
    categorical_transformer = make_pipeline(OneHotEncoder())
    
    # Numerical preprocessing pipeline
    numeric_features = sorted(numerical_features)
    numeric_transformer = make_pipeline(StandardScaler())

    # Put the 2 tracks together into one pipeline using the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop", 
    )

    # Get a list of the columns we used
    used_columns = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))

    # Append classifier to preprocessing pipeline.
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**model_params)),
        ]
    )
    return pipe, used_columns    

def get_output_transformer():
    '''return a labelbinarizer to transform the output labels'''
    return LabelBinarizer()


def train(df_path, categorical_features, numerical_features, model_params, output_label):
    """
    Train a ml based pipeline, logs the precision, recall, fbeta scores on train data
      and return the trained pipeline and also output transformer.

    Inputs
    ------
    df_path: path to the train data containing both input features and also output labels
    categorical_features (list): list of categorical features 
    numerical_features (list): list of numerical features 
    model_params (dict): ml model parameters 
    output_label (str): output label column name found in train data
    Returns
    -------
    model
        Trained pipeline.
    """

    df = pd.read_csv(df_path)
    
    y = df[output_label]

    input_pipe, params = get_training_inference_pipeline(categorical_features, numerical_features, model_params)
    output_pipe = get_output_transformer()

    X = df[params]
    y = output_pipe.fit_transform(y.values).ravel()
    
    input_pipe.fit(X, y)
    preds = input_pipe.predict(X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # log metrics
    mlflow.log_metrics({
        "train_precision": precision,
        "train_recall": recall,
        "train_fbeta": fbeta})

    return input_pipe, output_pipe


def export_model(input_pipe, output_pipe, model_save_path):
    '''save a dict of {"input":input_pipe, "output":output_pipe} in the give path'''
    saved_model = {"input":input_pipe, "output":output_pipe}
    with open(model_save_path, 'wb') as f:
        pickle.dump(saved_model, f)


def compute_model_metrics(y, preds):
    """
    Compute precision, recall, and F1.

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
    model (Pipeline): Trained machine learning pipeline.
    X (DataFrame): Data used for prediction.
    Returns
    -------
    preds (np.array): Predictions from the model.
    """

    preds = model.predict(X)
    return preds
