from typing import Optional

import mlflow
import pandas as pd


class Callback:
    def on_train_start(self, trainer_cv):
        pass

    def on_train_end(self, trainer_cv):
        pass

    def on_fold_start(self, trainer_cv):
        pass

    def on_fold_end(self, trainer_cv):
        pass


class MlflowCallback(Callback):
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        autolog: bool = True,
        start_run: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.autolog = autolog
        self.start_run = start_run

    def on_train_start(self, trainer_cv):
        # set experiment
        if self.experiment_name:
            mlflow.set_experiment(
                experiment_name=self.experiment_name,
            )

        if self.start_run:
            self.parent_run = mlflow.start_run()
        else:
            self.parent_run = mlflow.active_run()

        if self.autolog:
            mlflow.autolog()
        mlflow.set_tag("nest", "parent")

    def on_train_end(self, trainer_cv):
        metrics: list[dict[str, float]] = pd.json_normalize(
            trainer_cv.cv_results_
        ).to_dict(orient="records")

        df_cv_results = pd.json_normalize(trainer_cv.cv_results_)

        mlflow.log_metrics(df_cv_results.mean().add_prefix("mean_").to_dict())
        mlflow.log_metrics(df_cv_results.std().add_prefix("std_").to_dict())

        # if start run in this callback, then end the run here
        if self.start_run:
            mlflow.end_run()

    def on_fold_start(self, trainer_cv):
        mlflow.start_run(nested=True)
        mlflow.set_tag("nest", "child")

    def on_fold_end(self, trainer_cv):
        mlflow.end_run()
