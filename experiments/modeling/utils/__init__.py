import numpy as np
import pandas as pd


def make_submit_df_from_y_pred(y_pred: np.ndarray, index: pd.Index) -> pd.DataFrame:
    sample_submission_path = (
        "/workspaces/spaceship-titanic/data/inputs/sample_submission.csv"
    )
    df_sample_submission = pd.read_csv(sample_submission_path)

    df_submission = pd.DataFrame(
        y_pred,
        columns=["Transported"],
        index=index,
    )
    df_submission = df_submission.reindex(index=df_sample_submission["PassengerId"])
    return df_submission
