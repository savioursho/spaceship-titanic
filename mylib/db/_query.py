from typing import Optional

import sqlalchemy


def str_to_sqlalchemy(
    table_column_dict: dict[str, list[str]],
    schema: Optional[str] = None,
) -> dict[str, sqlalchemy.Table]:
    ret = dict()
    for table_name, columns in table_column_dict.items():
        table_obj = sqlalchemy.table(
            table_name,
            *(sqlalchemy.column(c) for c in columns),
            schema=schema,
        )
        ret[table_name] = table_obj
    return ret


def feature_joining_statement_from_dict(
    feature_table_column_dict: dict[str, list[str]],
    id_cols: list[str],
    schema: Optional[str] = None,
) -> sqlalchemy.Select:
    table_column_dict_with_id_cols = {
        table_name: columns + id_cols
        for table_name, columns in feature_table_column_dict.items()
    }
    tables = str_to_sqlalchemy(table_column_dict_with_id_cols, schema=schema)

    table_column_generator = (
        (table_name, column)
        for table_name, columns in feature_table_column_dict.items()
        for column in columns
    )
    table_generator = iter(tables.values())

    left_table = next(table_generator)

    # SELECT
    stmt = sqlalchemy.select(
        # id_cols
        *(left_table.c[id_col] for id_col in id_cols),
        # features
        *(
            tables[table_name].c[column]
            for table_name, column in table_column_generator
        ),
    )

    # FROM
    stmt = stmt.select_from(left_table)

    # JOIN
    for table in table_generator:
        stmt = stmt.join(
            table,
            sqlalchemy.and_(
                *(left_table.c[id_col] == table.c[id_col] for id_col in id_cols)
            ),
        )

    return stmt
