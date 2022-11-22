import numpy as np
import pytest
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder

# Feature engineering
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

from classification_model.config.core import config
from classification_model.processing.preprocess import Mapper

[100, 1_000, 2_000, -100]


def test_mappings_transformer(sample_test_input_data):
    """This tests the feature(s) generated using the mappings transformer."""
    # Given
    variables = config.model_config.mapping_vars
    mapper = Mapper(
        variables=variables,
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
    variables = config.model_config.numerical_vars_with_na
    missing_indicator = AddMissingIndicator(variables=variables)
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


def test_categorical_imputer_transformer(sample_train_input_data):
    """This tests the feature(s) generated using the CategoricalImputer transformer."""
    # Given
    variables = config.model_config.categorical_vars
    cat_imputer = CategoricalImputer(imputation_method="frequent", variables=variables)
    # No of NaNs in each column
    expected_trans_result = {
        "addr_state": 0,
        "application_type": 0,
        "emp_length": 0,
        "grade": 0,
        "home_ownership": 0,
        "initial_list_status": 0,
        "purpose": 0,
        "term": 0,
        "verification_status": 0,
        "zip_code": 0,
    }

    # When
    cat_imputer.fit(sample_train_input_data)  # Fit
    result = cat_imputer.transform(sample_train_input_data)  # Transform

    # Then
    result[variables].isna().sum().to_dict() == expected_trans_result


def test_categorical_encoder_transformer(sample_train_input_data):
    """This tests the feature(s) generated using the CategoricalImputer transformer."""
    # Given
    variables = config.model_config.categorical_vars
    cat_imputer = CategoricalImputer(imputation_method="frequent", variables=variables)
    ordinal_enc = OrdinalEncoder(encoding_method="ordered", variables=variables)
    # Impute NaNs
    cat_imputer.fit(sample_train_input_data)  # Fit
    df = cat_imputer.transform(sample_train_input_data)

    X = df[config.model_config.features]
    y = df[config.model_config.target]

    # No of NaNs in each column
    expected_trans_result = [
        {
            "addr_state": 29,
            "application_type": 0,
            "emp_length": 2,
            "grade": 4,
            "home_ownership": 3,
            "initial_list_status": 0,
            "purpose": 2,
            "term": 1,
            "verification_status": 1,
            "zip_code": 173,
        },
        {
            "addr_state": 17,
            "application_type": 0,
            "emp_length": 1,
            "grade": 2,
            "home_ownership": 2,
            "initial_list_status": 0,
            "purpose": 2,
            "term": 0,
            "verification_status": 2,
            "zip_code": 36,
        },
    ]

    # When
    ordinal_enc.fit(X, y)  # Fit
    trans_df = ordinal_enc.transform(X)  # Transform

    # Then
    # The encoded variables
    trans_df[variables].iloc[:2].to_dict(orient="records") == expected_trans_result


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_rare_label_encoder_transformer(sample_train_input_data):
    """This tests the feature(s) generated using the RareLabelEncoder transformer."""
    # Given
    variables = config.model_config.categorical_vars
    cat_imputer = CategoricalImputer(imputation_method="frequent", variables=variables)
    rare_label_enc = RareLabelEncoder(tol=0.05, n_categories=10, variables=variables)
    # Impute NaNs
    cat_imputer.fit(sample_train_input_data)  # Fit
    df = cat_imputer.transform(sample_train_input_data)

    X = df[config.model_config.features]
    y = df[config.model_config.target]

    # No of NaNs in each column
    expected_trans_result = {
        "addr_state": ["Rare", "TX", "FL", "NY", "CA"],
        "emp_length": [
            "2 years",
            "10+ years",
            "5 years",
            "Rare",
            "4 years",
            "< 1 year",
            "3 years",
            "1 year",
        ],
        "purpose": [
            "debt_consolidation",
            "home_improvement",
            "credit_card",
            "Rare",
            "other",
        ],
    }

    # When
    rare_label_enc.fit(X, y)  # Fit
    trans_df = rare_label_enc.transform(X)  # Transform

    # Then
    # The encoded "Rare" variables
    dict_ = {
        key: list(trans_df[key].unique())
        for key in ["addr_state", "emp_length", "purpose"]
    }
    dict_ == expected_trans_result


def test_median_imputer_transformer(sample_train_input_data):
    """This tests the feature(s) generated using the MedianImputer transformer."""
    # Given
    variables = config.model_config.repl_vars_with_median
    mean_imputer = MeanMedianImputer(imputation_method="median", variables=variables)
    # No of NaNs in each column
    expected_trans_result = {
        "all_util": 0,
        "dti": 0,
        "inq_last_6mths": 0,
        "mths_since_last_delinq": 0,
        "mths_since_last_major_derog": 0,
    }

    # When
    mean_imputer.fit(sample_train_input_data)  # Fit
    result = mean_imputer.transform(sample_train_input_data)  # Transform

    # Then
    result[variables].isna().sum().to_dict() == expected_trans_result


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("input", "expected"),
    [(100, 4.6052), (1_000, 6.9078), (2_000, 7.6009), (-100, np.nan)],
)
def test_log_transform(input, expected):
    """This tests the logarithm transformations."""
    # Given

    # When
    result = round(np.log(input), 4)
    # Then
    if not np.isnan(result):
        assert result == expected
    else:
        np.isnan(result)
