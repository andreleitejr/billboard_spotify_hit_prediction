import joblib
import pandas as pd
from src.utils.config import TEST_DATA_PATH, PREPROCESSOR_PATH, MODEL_PATH, PREDICTIONS_PATH
from src.scripts.data import load_data

def preprocess_test_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the test data using the training columns saved earlier."""
    data = pd.get_dummies(data, columns=['Genre'], drop_first=True)

    train_columns = joblib.load(PREPROCESSOR_PATH)

    for col in train_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[train_columns]

    return data


def test():
    """ Loads the test dataset, applies preprocessing, loads the trained model,
    generates predictions, maps prediction labels to human-readable values,
    and saves the results as a CSV file with 'Track Id' and 'Prediction' columns."""
    test_data = load_data(TEST_DATA_PATH)

    X_test = test_data.drop(['Peak Position', 'Hit', 'Song', 'Artist'], axis=1, errors='ignore')
    X_test_processed = preprocess_test_data(X_test)

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test_processed)

    label_map = {0: "Not Hit", 1: "Hit"}
    prediction_labels = [label_map[p] for p in predictions]

    results = pd.DataFrame({
        'Track Id': test_data['Song'],
        'Prediction': prediction_labels
    })

    results.to_csv(PREDICTIONS_PATH, index=False)

    print(f'✅ Previsões salvas em: {PREDICTIONS_PATH}')


if __name__ == '__main__':
    test()
