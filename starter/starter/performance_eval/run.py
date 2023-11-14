import argparse
import yaml
import sys
import itertools
sys.path.append('../')
from ml.model import *

import pandas as pd

# def get_sub_df_cat(df, feature, value):
#     sub_df = df[df['feature']==value]
#     return sub_df

# def get_sub_df_performnace(model, sub_df, categorical_features, label, feature, encoder, lb):
#     performances = {}

#     sub_X, sub_y, _, _ = process_data(
#         test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
#     )

def go(args):

    data_path =  args.data_path
    model_path = args.model_path
    output_label = args.output_label

    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    input_pipe, output_transformer = model_dict['input'], model_dict['output']

    eval_df = pd.read_csv(data_path)
    y = eval_df[output_label]
    y = output_transformer.transform(y)

    used_columns = list(itertools.chain.from_iterable([x[2] for x in input_pipe['preprocessor'].transformers]))
    X = eval_df[used_columns]

    preds = input_pipe.predict(X)



    # feature = args.categorical_feature
    # data_path = args.data_path
    # # Get the yaml file containing the parameters
    # with open(args.model_config) as fp:
    #     model_config = yaml.safe_load(fp)
    # cat_features = model_config['train']['cat_features']
    # model_path = model_config['model']['model_save_path']
    
    # # load in the data.
    # data_df = pd.read_csv(data_path)

    # for value in data_df[feature].unique():
    #     value_df = get_sub_df_cat(data_df, feature, value)
        


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
        type=str,
        nargs="+",
        help="list of categorical features for slice evaluation",
        required=False,
        default=None
    )

    args = parser.parse_args()

    go(args)
    