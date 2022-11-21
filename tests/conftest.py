import pytest
import pandas as pd
from classification_model.processing.data_manager import load_data
from classification_model.config.core import config


@pytest.fixture()
def sample_test_input_data() -> pd.DataFrame:
    """This returns the test data as a DF for testing."""
    data = load_data(filepath=config.app_config.test_data_file, is_train=False)
    return data

@pytest.fixture()
def sample_train_input_data() -> pd.DataFrame:
    """This returns the train data as a DF for testing."""
    data = load_data(filepath=config.app_config.sample_train_data_file, is_train=True)
    return data