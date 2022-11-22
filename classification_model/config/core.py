"""This module is used to parse and convert the YAML configuration to Python code."""

import typing as tp
from pathlib import Path

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# Project Directories
PACKAGE_ROOT = Path(classification_model.__file__).absolute().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    sample_train_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model training and feature engineering.
    """

    csv_format: str
    test_size: float
    random_state: int
    na_thresh: int
    target: str
    features: tp.List[str]
    num_vars_to_drop: tp.List[str]
    cat_vars_to_drop: tp.List[str]
    numerical_vars_with_na: tp.List[str]
    categorical_vars_with_na_frequent: tp.List[str]
    repl_vars_with_median: tp.List[str]
    repl_vars_with_mean: tp.List[str]
    log_transformed_vars: tp.List[str]
    yeo_johnson_transformed_vars: tp.List[str]
    numerical_vars_to_bin: tp.List[str]
    mapping_vars: tp.List[str]
    emp_length_mappings: tp.Dict[str, str]
    numerical_vars: tp.List[str]
    continuous_vars: tp.List[str]
    discrete_vars: tp.List[str]
    var_with_rare_labels: tp.List[str]
    categorical_vars: tp.List[str]


class Config(BaseModel):
    """Main config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file.

    Params:
    -------
    None
    Returns:
    --------
    None
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: tp.Union[Path, None] = None) -> YAML:
    """Parse and load YAML containing the package configuration.

    Params:
    -------
    cfg_path (Path): The configuration path.

    Returns:
    --------
    parsed_config (YAML): The YAML file containing the package configuraton.
    """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as cfg_file:
            parsed_config = load(cfg_file.read())
            return parsed_config
    # Else
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values.

    Params:
    -------
    parsed_config (YAML): The YAML file containing the package configuraton.

    Returns:
    --------
    _config (Config): The validated app and model configuration.
    """
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
