import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy
from typing_extensions import Self


def get_engine(
    sqlite_path: Optional[str] = None,
    echo: bool = False,
) -> sqlalchemy.Engine:
    sqlite_path = sqlite_path or os.environ["SQLITE_DB_PATH"]
    return create_engine_from_sqlite_path(
        sqlite_path=sqlite_path,
        echo=echo,
    )


def create_engine_from_sqlite_path(
    sqlite_path: str,
    echo: bool = False,
) -> sqlalchemy.Engine:
    return sqlalchemy.create_engine(f"sqlite:///{sqlite_path}", echo=echo)


def execute_query(
    connection: sqlalchemy.Connection, query: str
) -> sqlalchemy.CursorResult:
    return connection.execute(sqlalchemy.text(query))


def get_table_object(
    table_name: str,
    connection: sqlalchemy.Connection,
    schema: Optional[str] = None,
):
    table_obj = sqlalchemy.Table(
        table_name,
        sqlalchemy.MetaData(),
        schema=schema,
        autoload_with=connection,
    )
    return table_obj


def create_unique_index(
    table_name: str,
    index_cols: list[str],
    connection: sqlalchemy.Connection,
    schema: Optional[str] = None,
):
    table_obj = get_table_object(
        table_name=table_name, connection=connection, schema=schema
    )
    index_col_objs = [getattr(table_obj.columns, index_col) for index_col in index_cols]

    unique_index_name = f"ui_{table_name}"
    if schema is not None:
        unique_index_name += f"_{schema}"

    unique_index = sqlalchemy.Index(
        unique_index_name,
        *index_col_objs,
        unique=True,
    )
    unique_index.create(connection)

    return None


def pandas_to_sql_with_unique_index(
    df: pd.DataFrame,
    table_name: str,
    connection: sqlalchemy.Connection,
    index_cols: list[str],
    schema: Optional[str] = None,
):
    # インデックスに設定するカラムがデータフレームに存在するか確認
    for index_col in index_cols:
        assert index_col in df.columns, f"{index_col} is not in the dataframe."

    # データをDBに格納する
    df.to_sql(
        name=table_name,
        con=connection,
        schema=schema,
        if_exists="replace",
        index=False,
    )

    # ユニークインデックスを作成する
    create_unique_index(
        table_name,
        index_cols,
        connection,
        schema=schema,
    )

    return None


def join_schema_table(schema_table_tuple: tuple) -> str:
    """
    (schema, table) の形式のタプルを文字列に連結して返す。

    Examples:
        ("default", "table1") -> default.table1
        (None, "table1") -> table1
    """
    return ".".join(filter(None, schema_table_tuple))


def generate_db_column_list(engine: sqlalchemy.Engine) -> pd.DataFrame:
    insp = sqlalchemy.inspect(engine)
    multi_columns = insp.get_multi_columns()

    list_dfs = []
    for schema_table, columns in multi_columns.items():
        schema_table_joined = join_schema_table(schema_table)
        df = pd.DataFrame(columns)
        df["table"] = schema_table_joined
        list_dfs.append(df)

    df = pd.concat(list_dfs).reset_index(drop=True)

    return df


def attach_databases(databases: list[Path], connection):
    queries = (
        " ".join(
            [
                "ATTACH DATABASE",
                "'" + str(db) + "'",
                "AS",
                db.stem,
            ]
        )
        for db in databases
    )

    for query in queries:
        execute_query(
            connection=connection,
            query=query,
        )

    return None


@dataclass
class DataBase:
    db_dir: str
    main_db_name: str = "main"
    db_extension: str = "db"
    echo: bool = False

    @classmethod
    def setup(cls, db_dir: str, database_names: Optional[list[str]] = None) -> Self:
        database_names = database_names or list()
        databases = [
            Path(db_dir) / f"{name}.{cls.db_extension}" for name in database_names
        ]
        Path(db_dir).mkdir(exist_ok=True)
        db = cls(db_dir)
        with db.connect() as connection:
            if database_names:
                attach_databases(databases=databases, connection=connection)

        return db

    @property
    def databases(self):
        return list(Path(self.db_dir).rglob(f"*.{self.db_extension}"))

    @property
    def main_database(self):
        return Path(self.db_dir) / f"{self.main_db_name}.{self.db_extension}"

    @property
    def sub_databases(self):
        return [db for db in self.databases if db.stem != self.main_db_name]

    @property
    def engine(self):
        return create_engine_from_sqlite_path(self.main_database, echo=self.echo)

    @contextlib.contextmanager
    def connect(self):
        connection = self.engine.connect()
        attach_databases(self.sub_databases, connection=connection)
        yield connection
        connection.close()
