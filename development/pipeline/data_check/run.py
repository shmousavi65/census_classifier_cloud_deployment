import os
import mlflow
import argparse
import sys
import subprocess
from pathlib import Path


def go(args):
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    log_path = here / "log.log"

    # Resolve data_path relative to repo root if not absolute
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (repo_root / data_path).resolve()
    if not data_path.exists():
        raise SystemExit(
            f"[data_check] Dataset not found at: {data_path}\n"
            f"Did you run 'dvc pull' from the repo root?"
        )

    # Run pytest in the component dir so its local conftest.py is discovered
    cmd = [sys.executable, "-m", "pytest", "-s", "-vv", ".", "--data_path",
           str(data_path)]
    with log_path.open("w", encoding="utf-8") as f:
        _ = subprocess.run(cmd, cwd=here, stdout=f,
                           stderr=subprocess.STDOUT, check=False)

    mlflow.log_artifact(str(log_path))
    os.remove(log_path)


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
