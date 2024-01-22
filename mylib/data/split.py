import itertools
from typing import Iterator

import numpy as np
from numpy import ndarray
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    check_cv,
)
from sklearn.model_selection._split import _BaseKFold


def make_cv_split_train_val_test(cv: _BaseKFold):
    # validate
    cv = check_cv(cv)
    supported_cv_classes = [
        KFold,
        GroupKFold,
        StratifiedKFold,
        StratifiedGroupKFold,
    ]
    if not any(
        isinstance(cv, supported_cv_class)
        for supported_cv_class in supported_cv_classes
    ):
        raise NotImplementedError(
            f"{str(cv)} is not supported.\nSupported class is [{', '.join(c.__name__ for c in supported_cv_classes)}]"
        )

    # wrapper cv class
    class TriSplitWrapper(_BaseKFold):
        def __init__(self) -> None:
            super().__init__(
                cv.get_n_splits(),
                shuffle=cv.shuffle,
                random_state=cv.random_state,
            )

        def split(
            self,
            X,
            y=None,
            groups=None,
        ) -> Iterator[tuple[ndarray, ndarray, ndarray]]:
            return split_train_val_test(cv.split(X, y, groups))

        def __str__(self) -> str:
            return f"TriSplitWrapper({str(cv)})"

        def __repr__(self) -> str:
            return str(self)

    return TriSplitWrapper()


def split_train_val_test(splitter):
    splitter1, splitter2 = itertools.tee(splitter)
    splitter2 = itertools.cycle(splitter2)
    next(splitter2)
    for (train1, eval), (train2, test) in zip(splitter1, splitter2):
        train = np.intersect1d(train1, train2)
        yield train, eval, test
