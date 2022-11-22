from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer

# ML Algos
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config
from classification_model.processing import preprocess as pp

logistic_pipe = Pipeline(
    steps=[
        # ========== IMPUTATION ==========
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
        ),
        # Impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.repl_vars_with_median,
            ),
        ),
        # Impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_config.repl_vars_with_mean,
            ),
        ),
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.model_config.categorical_vars_with_na_frequent,
            ),
        ),
        # ========== TRANSFORM NUMERICAL VARIABLES ==========
        (
            "log_transform",
            LogTransformer(variables=config.model_config.log_transformed_vars),
        ),
        (
            "yeo_johnson_transform",
            YeoJohnsonTransformer(
                variables=config.model_config.yeo_johnson_transformed_vars
            ),
        ),
        # ========== DISCRETIZE NUMERICAL VARIABLES ==========
        (
            "discretizer",
            EqualFrequencyDiscretiser(
                variables=config.model_config.discrete_vars,
                q=10,
                return_object=True,
            ),
        ),
        # ========== CLEAN CATEGORICAL VARIABLES ==========
        (
            "emp_length_mapper",
            pp.Mapper(
                variables=config.model_config.mapping_vars,
                mappings=config.model_config.emp_length_mappings,
            ),
        ),
        # ========== REMOVE RARE LABELS ==========
        (
            "rare_labels",
            RareLabelEncoder(
                tol=0.05,
                n_categories=5,
                variables=config.model_config.var_with_rare_labels,
            ),
        ),
        # ========== ENCODE CATEGORICAL VARIABLES ==========
        (
            "categorical_encoder",
            OrdinalEncoder(
                encoding_method="ordered",
                variables=config.model_config.categorical_vars
                + config.model_config.discrete_vars,
            ),
        ),
        # ========== SCALE VARIABLES ==========
        (
            "scaler",
            StandardScaler(),
        ),
        # ========== LOGISTIC MODEL ==========
        (
            "log_model",
            LogisticRegression(random_state=config.model_config.random_state),
        ),
    ]
)

