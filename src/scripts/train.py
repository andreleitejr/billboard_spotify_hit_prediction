import argparse
from src.scripts.analysis import plot_confusion_matrix, numeric_histogram, categorical_count_plot, numeric_pair_plot, numeric_correlation_heatmap
from src.utils.config import TRAIN_DATA_PATH, MODEL_PATH
from src.scripts.data import load_data, split_data, preprocess_data, feature_engineering
from src.scripts.model import train_model, save_model, validate_model


def train(analysis: bool):
    """Runs the full training pipeline: loads and engineers data, splits into train/validation,
    preprocesses features, trains the model, validates performance, saves the model, and optionally
    performs exploratory analysis through visualizations if analysis=True."""
    data = load_data(TRAIN_DATA_PATH)

    if analysis:
        numeric_histogram(data)
        numeric_pair_plot(data)
        numeric_correlation_heatmap(data)
        data_filtered = data.drop(['Peak Position', 'Song', 'Artist'], axis=1)
        categorical_count_plot(data_filtered)

    data = feature_engineering(data)

    X_train, X_valid, y_train, y_valid = split_data(data)
    X_train_processed = preprocess_data(X_train)

    model = train_model(X_train_processed, y_train)
    validate_model(model, X_valid, y_valid, analysis=analysis)

    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--analysis', action='store_true', help='Enable analysis mode')
    args = parser.parse_args()

    train(analysis=args.analysis)
