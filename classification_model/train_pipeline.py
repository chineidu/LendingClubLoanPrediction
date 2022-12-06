# Custom modules
from config.core import config
from pipeline import logistic_pipe
from processing.data_manager import load_data, save_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics


def run_training() -> None:
    """Train the model."""

    # Read training data
    data = load_data(
        filepath=config.app_config.training_data_file,
        format_=config.model_config.csv_format,
        is_train=True,
    )
    # Divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # Set random seed for reproducibility
        random_state=config.model_config.random_state,
    )

    print("=============== Training The Pipeline ===============")
    # Fit model
    logistic_pipe.fit(X_train, y_train)

    # Evaluate Model
    print("=============== Evaluating The Pipeline ===============")
    train_auc = metrics.roc_auc_score(
        y_train, logistic_pipe.predict_proba(X_train)[:, 1]
    )
    test_auc = metrics.roc_auc_score(y_test, logistic_pipe.predict_proba(X_test)[:, 1])

    print(f"Train_auc_roc: {round(train_auc, 2)}\nTest_auc_roc: {round(test_auc, 2)}\n")

    # Persist trained model
    save_pipeline(pipeline_to_persist=logistic_pipe)


if __name__ == "__main__":
    run_training()
