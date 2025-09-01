import os
import mlflow
import argparse
import sys
import subprocess
from pathlib import Path


def go(args):
    log_file = "log.log"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        "-s", "-vv", ".",
        "--data_path", args.data_path,
    ]
    with log_path.open("w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           check=False)
    if os.getenv("DATA_CHECK_LOG_MLFLOW", "0") == "1":
        mlflow.log_artifact(str(log_path))
    # make CI fail if tests failed
    if result.returncode != 0:
        raise SystemExit(result.returncode)
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
