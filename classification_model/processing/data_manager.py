import typing as tp
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_data(
    *,
    filepath: str,
    format_: str = "csv",
    sheet_name: tp.Union[str, int, None] = 0,
    delimiter: tp.Union[str, None] = None,
    is_train: bool = True,
) -> pd.DataFrame:
    """This is used to load csv or excel data.

    Params:
    -------
    filepath (str): The filepath to the input data.
    format_ (str, default=csv): The format of the input data. It can be 'csv' or 'excel'.
    sheet_name (Union[str, int, None], default=0): The name of the excel sheet.
    delimiter (Union[str, None], default=None): The delimiter to use.
    is_train (bool, default=True): True if the data is the train_data otherwise False

    Returns:
    --------
    data: Pandas DF
        A DF containing the loaded input data.
    """
    low_memory = False
    data = (
        pd.read_csv(
            filepath_or_buffer=Path(f"{DATASET_DIR}/{filepath}"),
            low_memory=low_memory,
            delimiter=delimiter,
        )
        if format_ == config.model_config.csv_format
        else pd.read_excel(io=filepath, sheet_name=sheet_name)  # if format_ == 'excel'
    )
    default_list = [
        "Charged Off",
        "Late (31-120 days)",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    ]

    if is_train:
        # Convert the loan_status to int
        data[config.model_config.target] = data[config.model_config.target].apply(
            lambda status: 1 if status in default_list else 0
        )
        # Drop unnecessary variables
        data.drop(
            columns=config.model_config.num_vars_to_drop
            + config.model_config.cat_vars_to_drop,
            inplace=True,
        )

    return data


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.joblib"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: tp.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
