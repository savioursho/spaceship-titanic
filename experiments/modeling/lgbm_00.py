import gc
from logging import getLogger
from pathlib import Path

import hydra
import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier, register_logger
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer

from mylib.data import DataSetType
from mylib.db import DataBase, feature_joining_statement_from_dict
from mylib.log import Timer
from mylib.train.gbdt import LGBMTrainerCV, MlflowCallback
from mylib.transformer import Transformer

from .utils import make_submit_df_from_y_pred

load_dotenv()
logger = getLogger(__name__)
register_logger(logger)
timer = Timer(logger)


def coerce_columns_str(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームのカラム名の型をstrにする。

    sqlalchemyで読み込んだデータフレームのカラム名の型がstrではないことがある。
    このままだとsklearnのestimatorのfeature_names_in_が設定されないのでこの関数を使う。
    """
    return df.reindex(columns=[str(col) for col in df.columns])


def downcast_dtype(df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()
    df_ = df_.assign(**df_.select_dtypes("O").astype("category"))
    df_ = df_.assign(**df_.select_dtypes("number").astype("float32"))
    return df_


def preprocessors(name: str) -> Transformer:
    if name == "downcast_dtype":
        return FunctionTransformer(
            func=downcast_dtype,
            feature_names_out="one-to-one",
        )


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig):
    # setup
    with timer.measure("setup"):
        mlflow.set_experiment(cfg.experiment_name)
        mlflow_run = mlflow.start_run()
        run_name = mlflow_run.info.run_name
        ROOT_DIR = Path(cfg.root_dir)
        MODEL_DIR = Path(cfg.model_dir)
        SUBMIT_DIR = Path(cfg.submit_dir)
        db = DataBase(cfg.db_dir, echo=True)
        mlflow.log_param(
            "hydra_output_dir",
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        )
        id_cols: list[str] = OmegaConf.to_container(cfg.id_cols, resolve=True)
        target: dict[str, list[str]] = OmegaConf.to_container(cfg.target, resolve=True)

        features: dict[str, list[str]] = OmegaConf.to_container(
            cfg.feature, resolve=True
        )
        oof_pred_table_name = Path(__file__).stem

    # loading inputs
    with timer.measure("loading inputs"):
        select_features = feature_joining_statement_from_dict(
            features, id_cols, "train"
        )
        select_target = feature_joining_statement_from_dict(target, id_cols, "train")

        with db.connect() as connection:
            X = pd.read_sql(
                select_features,
                connection,
                index_col=id_cols,
            ).pipe(coerce_columns_str)
            y = (
                pd.read_sql(
                    select_target,
                    connection,
                    index_col=id_cols,
                )
                .pipe(coerce_columns_str)
                .iloc[:, 0]
                .astype(bool)
            )

    # model
    model_class_name = cfg.model.model_class
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model = globals()[model_class_name](**model_params)

    preprocessor = preprocessors(name=cfg.model.preprocessor)

    # cv
    cv_class_name = cfg.cv.cv_class
    cv_params = OmegaConf.to_container(cfg.cv.params, resolve=True)
    cv = globals()[cv_class_name](**cv_params)

    # scorers
    scorers = {name: metrics.get_scorer(name) for name in cfg.scorings}

    # training
    trainer = LGBMTrainerCV(
        estimator=model,
        preprocessor=preprocessor,
        scorers=scorers,
        cv=cv,
        callbacks=[MlflowCallback()],
    )

    trainer.fit(X, y)

    del X, y
    gc.collect()

    # dumping
    MODEL_RUN: str = "-".join([model_class_name, run_name])
    model_file_name = MODEL_RUN + ".pkl"
    model_file_path = MODEL_DIR / model_file_name
    joblib.dump(
        trainer,
        model_file_path,
        compress=True,
    )

    # loading test inputs

    select_features = feature_joining_statement_from_dict(features, id_cols, "test")

    with db.connect() as connection:
        X_test = pd.read_sql(
            select_features,
            connection,
            index_col=id_cols,
        ).pipe(coerce_columns_str)

    # predicting

    y_pred = trainer.predict(X_test)
    df_submission = make_submit_df_from_y_pred(y_pred, index=X_test.index)
    df_submission.to_csv(SUBMIT_DIR / f"{MODEL_RUN}.csv")


if __name__ == "__main__":
    main()
