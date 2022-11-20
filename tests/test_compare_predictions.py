import numpy as np
from classification_model.predict import make_predictions


def test_compare_predictions(sample_test_input_data):
    """This tests and compares a boolean array where two arrays are element-wise equal within a
    tolerance."""
    # Given
    # The expected probability of default for the 1st 10 predictions
    expected_pred_proba = [
        0.066,
        0.048,
        0.027,
        0.0,
        0.005,
        0.057,
        0.093,
        0.109,
        0.005,
        0.037,
    ]

    # When
    result = make_predictions(input_data=sample_test_input_data)
    pred_proba = result.get("default_probability")[:10]

    # Then
    assert np.isclose(expected_pred_proba, pred_proba, atol=0.03).all()
