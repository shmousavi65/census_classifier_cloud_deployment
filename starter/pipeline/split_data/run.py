import argparse, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

log_file = 'log.log'

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
    )
logger = logging.getLogger()

def go(args):

    data = pd.read_csv(args.raw_data_path)
    train, test = train_test_split(data, test_size=args.test_size, random_state=args.random_state)
    
    raw_dirname = os.path.dirname(args.raw_data_path)
    raw_basename = os.path.basename(args.raw_data_path)
    train_basename = raw_basename.split(".")[0] + "_train.csv"  
    test_basename = raw_basename.split(".")[0] + "_test.csv"  
    train_save_path = os.path.join(raw_dirname, train_basename)
    test_save_path = os.path.join(raw_dirname, test_basename)
    
    train.to_csv(train_save_path, index=False)
    test.to_csv(test_save_path, index=False)

    logger.info(f"splitted input data to train and test and save them in \n {train_save_path} \n {test_save_path} \n")
    mlflow.log_artifact(log_file)
    os.remove(log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data to train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--raw_data_path",
        type=str,
        help=" training data path",
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

    args = parser.parse_args()

    go(args)