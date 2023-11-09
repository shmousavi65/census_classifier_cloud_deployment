# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import *
import pandas as pd
import pickle
import argparse
import yaml

def go(args):
    # params

    # Get the configuration for the pipeline
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    model_params = model_config['train']['model_params']
    cat_features = model_config['train']['cat_features']

    data_path = args.data_path
    random_state = args.random_state
    test_size = args.test_size
    model_save_path = args.model_save_path

    # load in the data.
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=random_state)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train, model_params)

    # Save the model as a pickle file
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)


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
        "--model_config",
        type=str,
        help="path to the yaml file containing the required config for training",
        required=True,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Seed for the random number generator.",
        required=True,
        default=32
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=True,
        default=0.2
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        help=" path to save the model",
        required=True,
    )

    args = parser.parse_args()

    go(args)