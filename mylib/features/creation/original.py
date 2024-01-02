import os
from pathlib import Path

import pandas as pd

from mylib.features.creation.base import FeatureBase


class Original(FeatureBase):
    def create_train(self) -> pd.DataFrame:
        df_input_train = self.load_input_train()
        df_input_train = _camel_to_snake_original_features(
            df_input_train.drop(columns="Transported")
        )
        return df_input_train

    def create_test(self) -> pd.DataFrame:
        df_input_test = self.load_input_test()
        df_input_test = _camel_to_snake_original_features(df_input_test)
        return df_input_test


def _camel_to_snake_original_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "PassengerId": "passenger_id",
            "HomePlanet": "home_planet",
            "CryoSleep": "cryo_sleep",
            "Cabin": "cabin",
            "Destination": "destination",
            "Age": "age",
            "VIP": "vip",
            "RoomService": "room_service",
            "FoodCourt": "food_court",
            "ShoppingMall": "shopping_mall",
            "Spa": "spa",
            "VRDeck": "vr_deck",
            "Name": "name",
        }
    )


def _load_input_train():
    path = Path(os.environ["ROOT_DIR"]) / "data" / "inputs" / "train.csv"
    df_train = pd.read_csv(path)
    return df_train


def _load_input_test():
    path = Path(os.environ["ROOT_DIR"]) / "data" / "inputs" / "test.csv"
    df_test = pd.read_csv(path)
    return df_test


if __name__ == "__main__":
    feature_obj = Original(
        name="original",
        load_input_train=_load_input_train,
        load_input_test=_load_input_test,
    )

    feature_obj.create_feature()
