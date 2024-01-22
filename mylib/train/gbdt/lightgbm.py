import dataclasses
from logging import getLogger
from typing import Any, Callable, Literal, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMModel
from matplotlib.axes import Axes
from sklearn import metrics
from sklearn.base import BaseEstimator, check_is_fitted, clone, is_classifier
from sklearn.calibration import check_cv
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble._voting import _BaseVoting
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import _safe_indexing
from tqdm.auto import tqdm

from mylib.log import Timer
from mylib.train.gbdt._callbacks import Callback
from mylib.transformer import IdentityTransformer, Transformer

__all__ = [
    "LGBMTrainer",
    "LGBMTrainerCV",
    "LGBM_DEFAULT_PARAMS",
]

logger = getLogger(__name__)
timer = Timer(logger)

LGBM_DEFAULT_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.1,
    "subsample": 0.6,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "random_state": 0,
    "force_row_wise": True,
}


def multiple_safe_indexing(*arrays, indices):
    return [_safe_indexing(array, indices) for array in arrays]


@dataclasses.dataclass
class LGBMTrainer(BaseEstimator):
    estimator: LGBMModel
    preprocessor: Transformer = IdentityTransformer()
    scorers: Optional[
        dict[str, Callable[[BaseEstimator, pd.DataFrame, pd.Series], float]]
    ] = None

    def __post_init__(self):
        self._estimator_type = self.estimator._estimator_type

    def _sk_visual_block_(self):
        """for repr visualization"""
        pipeline = make_pipeline(
            self._clone_preprocessor(),
            self._clone_estimator(),
        )
        return pipeline._sk_visual_block_()

    def _clone_estimator(self) -> LGBMModel:
        return clone(self.estimator)

    def _clone_preprocessor(self) -> Transformer:
        return clone(self.preprocessor)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ):
        # check input
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)

        # preparing metrics
        scorers = self.scorers or dict(score=metrics.check_scoring(self.estimator))
        self.scorers_ = metrics._scorer._MultimetricScorer(scorers=scorers)

        # initializing preprocessor
        cloned_preprocessor = self._clone_preprocessor()

        # preprocessing train set
        train_set = (
            cloned_preprocessor.fit_transform(X, y),
            y,
        )

        # preprocessing test set if it's given
        if test_set is not None:
            test_set = (
                cloned_preprocessor.transform(test_set[0]),
                test_set[1],
            )

        # preparing fit_params
        fit_params = self._get_fit_params(
            train_set=train_set, test_set=test_set, **kwargs
        )

        # initializing estimator
        cloned_estimator = self._clone_estimator()

        # fitting estimator
        cloned_estimator.fit(**fit_params)

        # constructing pipeline and setting as attr
        self.pipeline_ = make_pipeline(
            cloned_preprocessor,
            cloned_estimator,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert is_classifier(self)
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        check_is_fitted(self)
        return self.scorers_(self.pipeline_, X, y)

    def _get_fit_params(
        self,
        train_set: tuple[pd.DataFrame, pd.Series],
        test_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ):
        eval_set = [train_set]
        eval_names = ["train"]
        if test_set is not None:
            eval_set.append(test_set)
            eval_names.append("test")

        n_estimators = self._clone_estimator().get_params()["n_estimators"]

        callbacks = [
            lgb.log_evaluation(period=max(1, n_estimators // 20)),
            lgb.early_stopping(
                stopping_rounds=n_estimators,
            ),
        ]

        fit_params = dict(
            X=train_set[0],
            y=train_set[1],
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks,
        )

        fit_params.update(kwargs)
        return fit_params

    def get_evals_result(self, metric: Optional[str] = None) -> pd.DataFrame:
        check_is_fitted(self)
        evals_result = self.pipeline_[1].evals_result_
        df_evals_result = pd.DataFrame().assign(
            **pd.json_normalize(evals_result).iloc[0]
        )
        df_evals_result.index.name = "Iterations"
        if metric:
            df_evals_result = df_evals_result.filter(like=metric)
        return df_evals_result

    def plot_evals_result(
        self, metric: Optional[str] = None, ax=None, **kwargs
    ) -> Axes:
        best_iteration = self.pipeline_[-1].best_iteration_

        ax = lgb.plot_metric(self.pipeline_[1], metric=metric, ax=ax, **kwargs)
        if best_iteration:
            ax.vlines(
                best_iteration,
                *ax.get_ylim(),
                colors="k",
                label=f"best iteration({best_iteration})",
                linestyles="dashed",
            )
            ax.legend()
        return ax

    def get_feature_importance(
        self,
        importance_type="gain",
        iteration=None,
    ):
        check_is_fitted(self)
        return (
            pd.DataFrame(
                {
                    "feature": self.pipeline_[1].feature_name_,
                    "importance": self.pipeline_[1].booster_.feature_importance(
                        importance_type=importance_type,
                        iteration=iteration,
                    ),
                }
            )
            .sort_values("importance", ascending=False)
            .set_index("feature")
        )

    def plot_feature_importance(
        self,
        importance_type="gain",
    ):
        check_is_fitted(self)
        return lgb.plot_importance(
            self.pipeline_[1],
            importance_type=importance_type,
        )

    def get_fitted_estimator(self) -> LGBMModel:
        check_is_fitted(self)
        return self.pipeline_[1]


@dataclasses.dataclass
class LGBMTrainerCV(LGBMTrainer, _BaseVoting):
    cv: Any = 5
    voting: str = "soft"
    weights: Any = None
    flatten_transform: bool = True
    callbacks: Optional[list[Callback]] = None

    def __post_init__(self):
        super().__post_init__()
        self.cv = check_cv(self.cv)

    def _clone_trainer(self):
        return LGBMTrainer(
            estimator=self.estimator,
            preprocessor=self.preprocessor,
            scorers=self.scorers,
        )

    def _execute_callbacks(
        self,
        stage=Literal["on_train_start", "on_train_end", "on_fold_start", "on_fold_end"],
    ):
        if self.callbacks:
            for callback in self.callbacks:
                getattr(callback, stage)(self)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        **kwargs,
    ):
        # preparing metrics
        scorers = self.scorers or dict(score=metrics.check_scoring(self.estimator))
        self.scorers_ = metrics._scorer._MultimetricScorer(scorers=scorers)

        # preparing for cv
        indices = self.cv.split(X=X, y=y, groups=groups)
        self.indices_ = list(indices)
        self.trainers_ = list()
        self.cv_results_ = list()
        preds = list()

        # running cv
        self._execute_callbacks(stage="on_train_start")

        for i, (train_idx, test_idx) in tqdm(
            enumerate(self.indices_), total=self.cv.get_n_splits()
        ):
            self._execute_callbacks(stage="on_fold_start")

            fold_results = dict()

            # splitting data
            train_set = X_train, y_train = multiple_safe_indexing(
                X, y, indices=train_idx
            )
            test_set = X_test, y_test = multiple_safe_indexing(X, y, indices=test_idx)

            # training
            with timer.measure(f"training fold{i}"):
                trainer = self._clone_trainer()
                trainer.fit(X_train, y_train, test_set, **kwargs)
                self.trainers_.append(trainer)
            fold_results["training_time"] = timer.duration

            # scoring
            with timer.measure(f"scoring fold{i}"):
                fold_results["test"] = trainer.score(*test_set)
                fold_results["train"] = trainer.score(*train_set)
                self.cv_results_.append(fold_results)
            fold_results["scoring_time"] = timer.duration

            # predicting
            if is_classifier(self):
                preds.append(trainer.predict_proba(X_test))
            else:
                preds.append(trainer.predict(X_test))

            self._execute_callbacks(stage="on_fold_end")

        if is_classifier(self):
            self.le_ = self.estimators_[0][-1]._le
            self.classes_ = self.le_.classes_
            self.preds_ = np.full(shape=(len(y), len(self.classes_)), fill_value=np.nan)
            columns = [f"proba_{c}" for c in self.classes_]
        else:
            self.preds_ = np.full(shape=len(y), fill_value=np.nan)
            columns = ["pred"]

        # organizing predictions
        _, test_indicies = zip(*self.indices_)
        test_indicies = np.concatenate(test_indicies)
        self.preds_[test_indicies] = np.concatenate(preds)
        self.preds_ = pd.DataFrame(self.preds_, index=X.index, columns=columns)

        self._execute_callbacks(stage="on_train_end")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if is_classifier(self):
            return VotingClassifier.predict(self, X)
        else:
            return VotingRegressor.predict(self, X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert is_classifier(self)
        return VotingClassifier.predict_proba(self, X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if is_classifier(self):
            return VotingClassifier.transform(self, X)
        else:
            return VotingRegressor.transform(self, X)

    def score(self, X: pd.DataFrame, y: pd.Series, folds: bool = False):
        check_is_fitted(self)
        if folds:
            scores = [trainer.score(X, y) for trainer in self.trainers_]
            return pd.json_normalize(scores).to_dict(orient="list")
        else:
            return self.scorers_(self, X, y)

    def _collect_probas(self, X: pd.DataFrame) -> np.ndarray:
        assert is_classifier(self)
        return VotingClassifier._collect_probas(self, X)

    @property
    def estimators_(self) -> list[Pipeline]:
        check_is_fitted(self)
        return [trainer.pipeline_ for trainer in self.trainers_]

    @property
    def estimators(self) -> list[Pipeline]:
        return [
            (f"fold{i}_", self.estimator) for i, trainer in enumerate(self.trainers_)
        ]

    def get_feature_names_out(self, input_features=None):
        if is_classifier(self):
            return VotingClassifier.get_feature_names_out(
                self, input_features=input_features
            )
        else:
            return VotingRegressor.get_feature_names_out(
                self, input_features=input_features
            )

    def _get_feature_importance_wide(self, importance_type="gain", iteration=None):
        def order_by_row_mean(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(index=df.mean(axis=1).sort_values(ascending=False).index)

        return pd.concat(
            [
                trainer.get_feature_importance(
                    importance_type, iteration=iteration
                ).add_suffix(f"_fold{i}")
                for i, trainer in enumerate(self.trainers_)
            ],
            axis=1,
        ).pipe(order_by_row_mean)

    def get_feature_importance(
        self,
        importance_type="gain",
        iteration=None,
        wide: bool = True,
    ):
        check_is_fitted(self)

        if wide:
            return self._get_feature_importance_wide(
                importance_type=importance_type, iteration=iteration
            )
        else:

            def to_long(df: pd.DataFrame) -> pd.DataFrame:
                return (
                    df.reset_index()
                    .melt(
                        id_vars="feature",
                        var_name="fold",
                        value_name="importance",
                    )
                    .assign(
                        fold=lambda df: df["fold"]
                        .str.replace("importance_fold", "")
                        .pipe(pd.to_numeric)
                    )
                    .reindex(columns=["fold", "feature", "importance"])
                )

            return self._get_feature_importance_wide(
                importance_type=importance_type, iteration=iteration
            ).pipe(to_long)

    def plot_feature_importance(self, importance_type="gain"):
        df_imp = self.get_feature_importance(
            importance_type=importance_type, wide=False
        )
        sns.catplot(
            df_imp,
            y="feature",
            x="importance",
            kind="bar",
            color="tab:blue",
            aspect=1.5,
            errorbar=None,
        ).set(title="Feature Importance")
        sns.stripplot(
            df_imp,
            y="feature",
            x="importance",
            color="tab:orange",
        )

    def plot_evals_result(self, metric: Optional[str] = None) -> None:
        fig, axes = plt.subplots(
            nrows=int(np.ceil(len(self.trainers_) / 2)),
            ncols=2,
            sharey=True,
            figsize=(10, 8),
        )
        axes = axes.ravel()
        fig.suptitle("Metric during training")
        for i, (trainer, ax) in enumerate(zip(self.trainers_, axes)):
            trainer.plot_evals_result(metric=metric, ax=ax, title=f"fold{i}")
        fig.tight_layout()
        return

    def get_cv_prediction(self, return_proba=False) -> pd.DataFrame:
        check_is_fitted(self)
        if return_proba:
            assert is_classifier(self)
            return self.preds_
        else:
            if is_classifier(self):
                return pd.DataFrame(
                    self.le_.inverse_transform(self.preds_.values.argmax(axis=1)),
                    index=self.preds_.index,
                    columns=["pred"],
                )
            else:
                return self.preds_
