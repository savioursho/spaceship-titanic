import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    ROOT_DIR = Path(os.environ["ROOT_DIR"])

    dirs_to_create = [
        ROOT_DIR / "data" / "outputs" / "models",
        ROOT_DIR / "data" / "outputs" / "submit",
    ]

    for dir in dirs_to_create:
        dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    main()
