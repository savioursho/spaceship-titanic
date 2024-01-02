import os
from pathlib import Path

import pandas as pd

from mylib.features.creation.base import FeatureBase


class Target(FeatureBase):
    def create_train(self) -> pd.DataFrame:
        df_input_train = self.load_input_train()
        df_input_train = df_input_train.loc[:, ["PassengerId", "Transported"]].rename(
            columns={
                "PassengerId": "passenger_id",
                "Transported": "transported",
            }
        )
        return df_input_train


def _load_input_train():
    path = Path(os.environ["ROOT_DIR"]) / "data" / "inputs" / "train.csv"
    df_train = pd.read_csv(path)
    return df_train


if __name__ == "__main__":
    feature_obj = Target(
        name="target",
        load_input_train=_load_input_train,
    )

    feature_obj.create_feature()
