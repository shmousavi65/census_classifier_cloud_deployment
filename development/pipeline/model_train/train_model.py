import logging
import argparse
import sys
import os
import ast
import mlflow

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from ml.model import train, export_model

log_file = 'log.log'

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger()


def go(args):

    # params
    model_params = args.model_params
    cat_features = args.categorical_features
    num_features = args.numerical_features
    data_path = args.data_path
    model_save_path = args.model_save_path
    output_label = args.output_label

    # create and train a pipeline
    logger.info("training the model ...")
    input_model, output_transformer = train(
        data_path, cat_features, num_features, model_params, output_label)

    # export the pipeline
    logger.info("exporting the input_model and output_transformer ...")
    export_model(input_model, output_transformer, model_save_path)

    logger.info("component run finished successfully!")

    mlflow.log_artifact(log_file)
    os.remove(log_file)


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

    parser.add_argument(
        "--output_label",
        type=str,
        help="data column name used as output label",
        required=True,
    )

    args = parser.parse_args()

    go(args)
