from enum import Enum, auto

from mylib.data.split import make_cv_split_train_val_test

__all__ = [
    "DataSetType",
    "make_cv_split_train_val_test",
]


class DataSetType(Enum):
    TRAIN = auto()
    TEST = auto()
