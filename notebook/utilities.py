"""This module contains helper functions for preprocessing."""

import pandas as pd
import typing as tp
import re
import itertools

# Helper Function(s)
def load_data(
    *,
    filepath: str,
    format_: str = "csv",
    sheet_name: tp.Union[str, int, None] = 0,
    delimiter: tp.Union[str, None] = None,
    low_memory: bool = True,
) -> pd.DataFrame:
    """This is used to load csv or excel data.

    Params:
    -------
    filepath: str
        The filepath to the input data.
    format_: str, default=csv
        The format of the input data. It can be 'csv' or 'excel'.
    sheet_name: Union[str, int, None], default=0
        The name of the excel sheet.
    delimiter: Union[str, None] default=None
        Delimiter to use. If sep is None, the C engine cannot automatically detect
    the separator, but the Python parsing engine can, meaning the latter will
    be used and automatically detect the separator by Python's builtin sniffer tool
    low_memory : bool, default=True
        Internally process the file in chunks, resulting in lower memory use
        while parsing, but possibly mixed type inference.  To ensure no mixed
        types either set False, or specify the type with the `dtype` parameter.
        Note that the entire file is read into a single DataFrame regardless,
        use the `chunksize` or `iterator` parameter to return the data in chunks.
        (Only valid with C parser).

    Returns:
    --------
    data: Pandas DF
        A DF containing the loaded input data.
    """
    data = (
        pd.read_csv(
            filepath_or_buffer=filepath, delimiter=delimiter, low_memory=low_memory
        )
        if format_ == "csv"
        else pd.read_excel(io=filepath, sheet_name=sheet_name)  # if format_ == 'excel'
    )
    data.columns = data.columns.str.strip()
    print(f"The shape of the data: {data.shape}\n")
    return data


def obtain_mapped_variables(
    *, data: pd.DataFrame, input_list: tp.List, name: str
) -> tp.List:
    """This is used to obtain the enum variable names.

    Params:
    -------
    data (Pandas DF): DF containing the input data. The column names will be
                      gotten from the DF.
    input_list (List): List containing the raw/original variable names.
    name (str): The name of the enum class.

    Returns:
    --------
    result (List): List containing the transformed emum variable names.
    """

    repl = ""
    
    if not isinstance(input_list, list):
        raise ValueError(f"{input_list} should be a list")
    raw_var_lower = sorted(data.columns)
    raw_var_upper = [var.upper() for var in raw_var_lower]
    # Dict containing the original and enum variable names as key-value pairs.
    var_dict = dict(itertools.zip_longest(raw_var_lower, raw_var_upper))
    enum_variables = []

    for var in input_list:
        if var in var_dict.keys():  # If the variable is in var_dict
            enum_variables.append(f"{name}.{var_dict.get(var)}")

    # Remove the quotes
    result = re.sub(pattern=r"['']", repl="", string=str(enum_variables))
    return result
