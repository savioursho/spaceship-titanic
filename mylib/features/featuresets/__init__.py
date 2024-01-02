from typing import Literal, Optional, Union

import pandas as pd
import sqlalchemy

from mylib.db import get_engine, get_table_object


def union_set_list(a: list, b: list) -> list:
    """
    リストの順序を保ったまま和集合を計算する
    https://qiita.com/tellko/items/585511f8b9973e48fc11
    """
    return list(dict.fromkeys(a + b))


def get_dataset(
    columns_dict: dict[str, list[str]],
    train_test: Literal["train", "test"],
    engine: Optional[sqlalchemy.Engine] = None,
) -> Union[tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    # engineが与えられなかったら、環境変数のパスからengineを作る
    engine = engine or get_engine()

    id_cols = ["passenger_id"]

    X = get_feature_set_from_dict(
        columns_dict=columns_dict,
        id_cols=id_cols,
        train_test=train_test,
        engine=engine,
    )

    if train_test == "test":
        return X

    y = get_feature_set_from_dict(
        columns_dict={"target": ["transported"]},
        id_cols=id_cols,
        train_test=train_test,
        engine=engine,
    ).iloc[:, 0]

    return X, y


def load_features_from_table(
    table_name: str,
    connection: sqlalchemy.Connection,
    cols: list[str],
    id_cols: list[str],
):
    table_obj = get_table_object(table_name=table_name, connection=connection)
    statment = sqlalchemy.select(table_obj.c[tuple(union_set_list(id_cols, cols))])
    df = pd.read_sql(statment, con=connection)
    df.set_index(id_cols, inplace=True)
    return df


def get_feature_set_from_dict(
    columns_dict: dict[str, list[str]],
    id_cols: list[str],
    train_test: Literal["train", "test"],
    engine: sqlalchemy.Engine,
) -> pd.DataFrame:
    list_dfs = []
    with engine.connect() as connection:
        for table, cols in columns_dict.items():
            table_name = table + "_" + train_test

            df = load_features_from_table(
                table_name=table_name,
                connection=connection,
                cols=cols,
                id_cols=id_cols,
            )

            list_dfs.append(df)
    df = pd.concat(list_dfs, axis=1)
    return df
