import sys
import subprocess
from pathlib import Path


def go():
    log_file = "log.log"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-s", "-vv", "."]
    with log_path.open("w", encoding="utf-8") as f:
        _ = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           check=False)


if __name__ == "__main__":
    go()
