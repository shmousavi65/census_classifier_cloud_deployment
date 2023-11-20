import argparse
import sys, os
import itertools
import logging
import ast
sys.path.append('../')
from ml.model import *
import pandas as pd

log_file = 'log.log'

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
    )
logger = logging.getLogger()

def get_sub_df_cat(df, feature, value):
    sub_df = df[df[feature]==value]
    return sub_df

def get_performnace_on_df(X_df, y_series, input_pipeline, output_transformer):
    y = output_transformer.transform(y_series)
    preds = inference(input_pipeline, X_df)
    return compute_model_metrics(y, preds)

def go(args):

    data_path =  args.data_path
    model_path = args.model_path
    output_label = args.output_label
    slice_eval_features = list(args.slice_eval_features)

    logger.info("loading the input_model and output_transformer ...")
    input_pipe, output_transformer = load_model(model_path)

    eval_df = pd.read_csv(data_path)
    
    y_series = eval_df[output_label]

    used_columns = list(itertools.chain.from_iterable([x[2] for x in input_pipe['preprocessor'].transformers]))
    X_df = eval_df[used_columns]

    # performance on entire data
    logger.info("calculating the model performance on the entire given data ...")
    precision, recall, fbeta = get_performnace_on_df(X_df, y_series, input_pipe, output_transformer)    
    mlflow.log_metrics({
        "test_precision": precision,
        "test_recall": recall,
        "test_fbeta": fbeta})
    
    logger.info("calculating the model performance on the slices of the given feature(s) ...")
    if slice_eval_features:
        for feature in slice_eval_features:
            vals = eval_df[feature].unique()
            for val in vals:
                sub_df = get_sub_df_cat(eval_df, feature, val)
                sub_y_series = sub_df[output_label]
                sub_X_df = sub_df[used_columns]
                precision, recall, fbeta = get_performnace_on_df(sub_X_df, sub_y_series, input_pipe, output_transformer)    
                mlflow.log_metrics({
                    f"test_{str.strip(feature)}_{str.strip(val)}_precision": precision,
                    f"test_{str.strip(feature)}_{str.strip(val)}_recall": recall,
                    f"test_{str.strip(feature)}_{str.strip(val)}_fbeta": fbeta})


    logger.info("component run finished successfully!")

    mlflow.log_artifact(log_file)
    os.remove(log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="calculate the performance on slices",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help=" test data for evaluation",
        required=True,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model for performance evaluation",
        required=True,
    )

    parser.add_argument(
        "--output_label",
        type=str,
        help="data column name used as output label",
        required=True,
    )

    parser.add_argument(
        "--slice_eval_features",
        type=ast.literal_eval,
        help="list of categorical features for slice evaluation",
        required=False,
        default=None
    )

    args = parser.parse_args()

    go(args)
    