import os
import mlflow
import argparse


def go(args):
    log_file = "log.log"
    exec_command = f"pytest -s -vv . --data_path {args.data_path} > {log_file}"
    os.system(exec_command)
    mlflow.log_artifact(log_file)
    os.remove(log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help=" data_path for test",
        required=True,
    )
    args = parser.parse_args()
    go(args)
