import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from src.scripts.analysis import plot_confusion_matrix


def load_model(model_path: str) -> RandomForestClassifier:
    """Loads a trained model from a file using joblib."""
    model = joblib.load(model_path)
    return model


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Trains an RandomForestClassifier with predefined hyperparameters."""

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


def validate_model(model, X_valid, y_valid, analysis=False) -> None:
    """Calculates the Mean Absolute Error (MAE) and Accuracy."""
    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    accuracy = accuracy_score(y_valid, predictions)

    print(
        f'✅ Validation completed!\n'
        f'📊 MAE (Cross-Validation): {mae:.2f} (± {mae:.2f})\n'
        f'🔍 Accuracy: {accuracy:.4f} ({accuracy:.0%})\n'
    )

    if analysis:
        plot_confusion_matrix(y_valid, predictions)


def save_model(model, model_path: str) -> None:
    """Saves a trained model to a file using joblib."""
    joblib.dump(model, model_path)
