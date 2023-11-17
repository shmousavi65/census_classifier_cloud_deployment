import logging, sys
import numpy as np
import pandas as pd

sys.path.append('../')
from ml.model import *

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def test_get_output_transformer():
    '''assert the output transformer is labelbinarizer'''
    print("HELELELELLELELELELLELEL")
    assert (get_output_transformer(), LabelBinarizer)

def test_compute_model_metrics():
    '''assert the performance scores are computed correctly'''
    dummy_y = np.ones((10,1))
    dummy_preds = np.ones((10,1))
    precision, recall, fbeta = compute_model_metrics(dummy_y, dummy_preds)
    assert (precision, recall, fbeta) == (1, 1 ,1), "The computed scores are not crrect!"

def test_get_training_inference_pipeline():
    '''assert the pipeline used columns functions correctly 
        (also sorted properly) and the classifier is LogisticRegression'''
    categorical_features = ['education', 'sex'] 
    numerical_features = ['score', 'age']
    model_params = {'C':101}
    pipe, used_columns = get_training_inference_pipeline(categorical_features, numerical_features, model_params)
    assert used_columns == ['age', 'score', 'education', 'sex']
    assert isinstance(pipe["classifier"], LogisticRegression)