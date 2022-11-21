import numpy as np

# Feature engineering
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder,
)

from classification_model.predict import make_predictions
from classification_model.processing.preprocess import Mapper
from classification_model.config.core import config

import typing as tp


def test_mappings_transformer(sample_test_input_data):
    """This tests the feature(s) generated using the mappings transformer."""
    # Given
    mapper = Mapper(
        variables=config.model_config.mapping_vars,
        mappings=config.model_config.emp_length_mappings,
    )
    expected_mappings = ["5", "9", "4", "0", "6", "10", "10"]
    # When
    result = mapper.fit_transform(sample_test_input_data)

    # Then
    assert result["emp_length"].iloc[3:10].to_list() == expected_mappings


def test_add_missing_indicator_transformer(sample_test_input_data):
    """This tests the feature(s) generated using the mappings transformer."""
    # Given
    missing_indicator = AddMissingIndicator(
        variables=config.model_config.numerical_vars_with_na
    )
    expected_trans_result = [
        {"il_util_na": 0, "mths_since_last_delinq_na": 0},
        {"il_util_na": 1, "mths_since_last_delinq_na": 0},
        {"il_util_na": 1, "mths_since_last_delinq_na": 0},
        {"il_util_na": 0, "mths_since_last_delinq_na": 1},
        {"il_util_na": 1, "mths_since_last_delinq_na": 1},
        {"il_util_na": 0, "mths_since_last_delinq_na": 0},
        {"il_util_na": 1, "mths_since_last_delinq_na": 1},
        {"il_util_na": 0, "mths_since_last_delinq_na": 0},
        {"il_util_na": 0, "mths_since_last_delinq_na": 1},
    ]

    # When
    missing_indicator.fit(sample_test_input_data)  # Fit
    result = missing_indicator.transform(sample_test_input_data)  # Transform

    # Then
    assert (
        result[["il_util_na", "mths_since_last_delinq_na"]]
        .iloc[:9]
        .to_dict(orient="records")
        == expected_trans_result
    )
