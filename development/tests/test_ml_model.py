import logging
import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
current_dir = os.path.dirname(os.path.abspath(__file__))
prev_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(prev_dir, "pipeline"))
from ml.model import compute_model_metrics, get_output_transformer, \
    get_training_inference_pipeline, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_get_output_transformer():
    '''assert the output transformer is labelbinarizer'''
    assert isinstance(get_output_transformer(), LabelBinarizer)


def test_compute_model_metrics():
    '''assert the performance scores are computed correctly'''
    dummy_y = np.ones((10, 1))
    dummy_preds = np.ones((10, 1))
    precision, recall, fbeta = compute_model_metrics(dummy_y, dummy_preds)
    assert (precision, recall, fbeta) == (
        1, 1, 1), "The computed scores are not crrect!"


def test_get_training_inference_pipeline():
    '''assert the pipeline used columns functions correctly
        (also sorted properly) and the classifier is LogisticRegression'''
    categorical_features = ['education', 'sex']
    numerical_features = ['score', 'age']
    model_params = {'C': 101}
    pipe, used_columns = get_training_inference_pipeline(
        categorical_features, numerical_features, model_params)
    assert used_columns == ['age', 'score', 'education', 'sex']
    assert isinstance(pipe["classifier"], LogisticRegression)


def test_load_model():
    path_to_model = os.path.join(prev_dir, "model/model.pkl")
    input_pipe, output_transformer = load_model(path_to_model)
    assert isinstance(output_transformer, LabelBinarizer)