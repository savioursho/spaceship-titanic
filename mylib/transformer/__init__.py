from typing import Protocol

from sklearn.preprocessing import FunctionTransformer


class Transformer(Protocol):
    def fit():
        ...

    def transform():
        ...

    def fit_transform():
        ...


class IdentityTransformer(FunctionTransformer):
    def __init__(self):
        super().__init__()
