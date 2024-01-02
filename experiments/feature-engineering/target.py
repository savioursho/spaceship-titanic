import gc
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Optional

import hydra
import mlflow
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from ydata_profiling import ProfileReport

from mylib.data import DataSetType
from mylib.db.core import DataBase, pandas_to_sql_with_unique_index
from mylib.log import Timer

load_dotenv()
logger = getLogger(__name__)
timer = Timer(logger)


def log_profile_to_mlflow(profile: ProfileReport, file_name: Optional[str] = None):
    file_name = file_name or "profile.html"

    with tempfile.TemporaryDirectory(dir=".") as tempdir:
        profile_path = Path(tempdir) / file_name
        profile.to_file(profile_path)
        mlflow.log_artifact(profile_path)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig):
    # setup
    with timer.measure("setup"):
        ROOT_DIR = Path(cfg.root_dir)
        db = DataBase(cfg.db_dir, echo=True)
        mlflow.set_experiment(cfg.experiment_name)
        mlflow.log_param(
            "hydra_output_dir",
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        )
        csv_path = {
            DataSetType.TRAIN: ROOT_DIR / "data/inputs/train.csv",
            DataSetType.TEST: ROOT_DIR / "data/inputs/test.csv",
        }
        id_cols: list[str] = OmegaConf.to_container(cfg.id_cols, resolve=True)
        feature_table_name = Path(__file__).stem

    # train set
    # load inputs
    with timer.measure("load inputs train"):
        df_input_train = pd.read_csv(csv_path[DataSetType.TRAIN])

    # create features
    with timer.measure("create feature train"):
        df_feature_train = df_input_train.loc[
            :,
            [
                "PassengerId",
                "Transported",
            ],
        ].rename(
            columns={
                "PassengerId": "passenger_id",
                "Transported": "transported",
            }
        )

        del df_input_train
        gc.collect()

    # profile
    with timer.measure("profile feature train"):
        profile_train = ProfileReport(
            df_feature_train.set_index(id_cols),
            title="train",
        )

    # dump features
    with timer.measure("dump feature train"):
        with db.connect() as connection:
            pandas_to_sql_with_unique_index(
                df=df_feature_train,
                table_name=feature_table_name,
                connection=connection,
                index_cols=id_cols,
                schema="train",
            )
            connection.commit()
        del df_feature_train
        gc.collect()

    with timer.measure("dump profile"):
        log_profile_to_mlflow(profile_train)

    mlflow.log_artifacts(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        "hydra_outputs",
    )


if __name__ == "__main__":
    main()
