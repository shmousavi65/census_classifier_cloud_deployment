import sys
import ast
sys.path.append('../')
from ml.model import *
import pandas as pd
import argparse
import yaml

def go(args):
    
    # params
    model_params = args.model_params
    cat_features = args.categorical_features
    num_features = args.numerical_features
    data_path = args.data_path
    model_save_path = args.model_save_path
    
    # create and train a pipeline
    input_model, output_transformer = train(data_path, cat_features, num_features, model_params, "salary")

    # export the pipeline
    export_model(input_model, output_transformer, model_save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression Model",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help=" training data path",
        required=True,
    )

    parser.add_argument(
        "--numerical_features",
        type=ast.literal_eval,
        help=" list of numerical features",
        required=True,
    )
    
    parser.add_argument(
        "--categorical_features",
        type=ast.literal_eval,
        help=" list of categorical features",
        required=True,
    )

    parser.add_argument(
        "--model_params",
        type=ast.literal_eval,
        help=" dict of model parameters",
        required=True,
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        help=" path to save the model",
        required=True,
    )

    args = parser.parse_args()

    go(args)