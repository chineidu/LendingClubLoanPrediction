"""This module contains helper classes used for preprocessing."""

import typing as tp

from sklearn.base import BaseEstimator, TransformerMixin


class Mapper(BaseEstimator, TransformerMixin):
    """This is used to map the variables using the mappings.

    Params:
    -------
    variables (List): The list containing the variables to map.
    mappings (Dict): The dict containing the key-value pairs used in the mapping.

    Attributes:
    -----------
    variables_: Get the list containing the variables to map.
    mappings_: Get the dict containing the key-value pairs used in the mapping.

    Methods:
    --------
    fit: Needed to accomodate the sklearn pipeline.
    transform: Map the variables.
    fit_transform: Fit and transform the data.
    """

    def __init__(self, *, variables: tp.List, mappings: tp.Dict):

        if not isinstance(variables, list):
            raise ValueError(f"{variables} should be a list")

        self.variables_ = variables
        self.mappings_ = mappings

    @property
    def variables(self):
        return self.variables_

    @property
    def mappings(self):
        return self.mappings_

    @variables.setter  # type: ignore
    def variables(self, value: tp.List):
        if isinstance(value, tp.List):
            self.variables_ = value
        else:
            raise Exception(f"{value} must be a list")

    @mappings.setter  # type: ignore
    def mappings(self, value: tp.Dict):
        if isinstance(value, tp.Dict):
            self.mappings_ = value
        else:
            raise Exception(f"{value} must be a dict")

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables_:
            X[feature] = X[feature].map(self.mappings_)

        return X
