import os
import mlflow
import argparse


def go():
    log_file = "log.log"
    exec_command = f"pytest -s -vv . >  {log_file}"
    os.system(exec_command)

if __name__ == "__main__":
    go()