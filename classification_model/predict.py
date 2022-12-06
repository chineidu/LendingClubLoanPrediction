# Builtin modules/packages
import typing as tp

import pandas as pd

from classification_model import __version__ as _version

# Custom Imports
from classification_model.config.core import config
from classification_model.processing.data_manager import load_data, load_pipeline
from classification_model.processing.validate import validate_inputs


def make_predictions(*, input_data: pd.DataFrame) -> tp.Dict:
    """This is used to calculate the probability of default and the
    loan_status.

    Params:
    -------
    input_data (Pandas DF): DF containing the input data.

    Returns:
    --------
    results (Dict): A dict containing the probability of default,
            default_status, model_version and the possible errors.
    """
    input_data = input_data.copy()
    file_name = f"{config.app_config.pipeline_save_file}{_version}.joblib"
    logistic_pipe = load_pipeline(file_name=file_name)
    # Validate input_data
    validated_data, errors = validate_inputs(input_data=input_data)
    result = {
        "default_probability": None,
        "default_status": None,
        "model_version": _version,
        "errors": errors,
    }

    if not errors:
        default_status = list(logistic_pipe.predict(validated_data))
        default_status = list(map(lambda x: "Yes" if x == 1 else "No", default_status))
        pred_proba = list(logistic_pipe.predict_proba(validated_data)[:, 1])
        pred_proba = [round(val, 3) for val in pred_proba]  # Round the values

        result = {
            "default_probability": pred_proba,  # type: ignore
            "default_status": default_status,  # type: ignore
            "model_version": _version,
            "errors": errors,
        }
    return result


