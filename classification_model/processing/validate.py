# Builtin modules/packages
import typing as tp

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

# Custom Imports
from classification_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """This is used to drop the NaNs that were not present in the training data.

    Params:
    -------
    input_data (Pandas DF): DF containing the input data.

    Returns:
    --------
    cleaned_data (Pandas DF): DF containing the validated data.
    """
    cleaned_data = input_data.copy()

    # Variables with NaNs in the training data
    vars_with_na_train_data = (
        config.model_config.numerical_vars_with_na
        + config.model_config.categorical_vars_with_na_frequent
    )

    vars_with_na_test_data = [
        var
        for var in cleaned_data.columns
        if cleaned_data[var].isna().sum() > config.model_config.na_thresh
    ]
    # Variables with NaN values in ONLY the test data.
    vars_with_na_test_ONLY = [
        *set(vars_with_na_test_data).difference(set(vars_with_na_train_data))
    ]
    # Drop the NaN values
    cleaned_data.dropna(subset=vars_with_na_test_ONLY, inplace=True)
    return cleaned_data


def validate_inputs(
    *, input_data: pd.DataFrame
) -> tp.Tuple[pd.DataFrame, tp.Optional[dict]]:
    """This is used to validate the input_data using a Pydantic Schema.

    Params:
    -------
    input_data (Pandas DF): DF containing the input data.

    Returns:
    --------
    validated_data (Pandas DF): DF containing the validated data.
    """
    input_data = input_data.copy()
    # Drop NaN values
    validated_data = drop_na_inputs(input_data=input_data)
    errors = None

    # Validate inputs. Replace NaN with None so that Pydantic can interpret it
    try:
        _ = ValidateLendingData(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as err:
        errors = err.json(indent=1)

    return (validated_data, errors)


class LendingDataSchema(BaseModel):
    """This is used to validate the data."""

    acc_now_delinq: tp.Optional[int]
    addr_state: tp.Optional[str]
    all_util: tp.Optional[float]
    annual_inc: tp.Optional[float]
    application_type: tp.Optional[str]
    collections_12_mths_ex_med: tp.Optional[int]
    delinq_2yrs: tp.Optional[int]
    dti: tp.Optional[float]
    emp_length: tp.Optional[str]
    grade: tp.Optional[str]
    home_ownership: tp.Optional[str]
    il_util: tp.Optional[float]
    initial_list_status: tp.Optional[str]
    inq_fi: tp.Optional[float]
    inq_last_12m: tp.Optional[float]
    inq_last_6mths: tp.Optional[float]
    int_rate: tp.Optional[float]
    last_pymnt_amnt: tp.Optional[float]
    loan_amnt: tp.Optional[int]
    max_bal_bc: tp.Optional[float]
    mths_since_last_delinq: tp.Optional[float]
    mths_since_last_major_derog: tp.Optional[float]
    mths_since_rcnt_il: tp.Optional[float]
    open_acc: tp.Optional[int]
    open_acc_6m: tp.Optional[float]
    open_il_12m: tp.Optional[float]
    open_il_24m: tp.Optional[float]
    open_rv_12m: tp.Optional[float]
    out_prncp: tp.Optional[float]
    pub_rec: tp.Optional[int]
    purpose: tp.Optional[str]
    recoveries: tp.Optional[float]
    revol_bal: tp.Optional[float]
    revol_util: tp.Optional[float]
    term: tp.Optional[str]
    tot_coll_amt: tp.Optional[float]
    tot_cur_bal: tp.Optional[float]
    total_acc: tp.Optional[int]
    total_bal_il: tp.Optional[float]
    total_cu_tl: tp.Optional[float]
    total_pymnt: tp.Optional[float]
    total_rec_int: tp.Optional[float]
    total_rec_late_fee: tp.Optional[float]
    total_rev_hi_lim: tp.Optional[float]
    verification_status: tp.Optional[str]
    zip_code: tp.Optional[str]


class ValidateLendingData(BaseModel):
    inputs: tp.List[LendingDataSchema]
