import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol

import pandas as pd
from dotenv import load_dotenv

from mylib.db import (
    create_engine_from_sqlite_path,
    generate_db_column_list,
    pandas_to_sql_with_unique_index,
)

load_dotenv()

db_path = os.environ["SQLITE_DB_PATH"]
engine = create_engine_from_sqlite_path(db_path)


def _update_column_list():
    column_list_path = Path(os.environ["ROOT_DIR"]) / "data/db/column_list.csv"
    df_column_list = generate_db_column_list(engine)
    df_column_list.to_csv(column_list_path, index=False)


class Transformer(Protocol):
    def fit(self, X, y=None, **fit_params):
        ...

    def transform(self, X, y=None, **fit_params):
        ...

    def fit_transform(self, X, y=None, **fit_params):
        ...


@dataclass
class FeatureBase:
    name: str
    load_input_train: Callable[[], pd.DataFrame]
    load_input_test: Optional[Callable[[], pd.DataFrame]] = None
    transformer: Optional[Transformer] = None
    transformer_path: Optional[str] = None

    def create_train(self) -> pd.DataFrame:
        raise NotImplementedError

    def create_test(self) -> pd.DataFrame:
        raise NotImplementedError

    def create_feature(self):
        # train
        df_feature_train = self.create_train()
        self.dump_df(
            df_feature_train,
            "train",
        )
        del df_feature_train

        # test
        if self.load_input_test is not None:
            df_feature_test = self.create_test()
            self.dump_df(
                df_feature_test,
                "test",
            )
            del df_feature_test

        # update column list
        _update_column_list()

    def dump_df(
        self,
        df: pd.DataFrame,
        train_test: Literal["train", "test"],
    ):
        assert train_test in ["train", "test"]

        table_name = self.name + "_" + train_test

        index_cols = ["passenger_id"]

        with engine.connect() as connection:
            pandas_to_sql_with_unique_index(
                df=df,
                table_name=table_name,
                connection=connection,
                index_cols=index_cols,
            )

        return None
