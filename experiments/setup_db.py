import os
from pathlib import Path

from dotenv import load_dotenv

from mylib.db.core import DataBase

load_dotenv()


def main():
    sqlite_db_dir = Path(os.environ["SQLITE_DB_DIR"])

    db = DataBase.setup(sqlite_db_dir, database_names=["train", "test"])


if __name__ == "__main__":
    main()
