import pandas as pd
from src.utils.config import TEST_DATA_PATH, PREPROCESSOR_PATH, MODEL_PATH, PREDICTIONS_PATH
from src.scripts.data import feature_engineering, load_data
from src.scripts.model import load_model


def test():
    """Loads a trained model and preprocessor, applies feature engineering to test data,
    transforms features, makes predictions, and saves the results to a CSV file."""
    test_data = load_data(TEST_DATA_PATH)

    preprocessor = load_model(PREPROCESSOR_PATH)
    test_data_transformed = preprocessor.transform(test_data)

    model = load_model(MODEL_PATH)
    predictions = model.predict(test_data_transformed)

    results = pd.DataFrame({'Id': test_data.index, 'SalePrice': predictions})
    results.to_csv(PREDICTIONS_PATH, index=False)


if __name__ == '__main__':
    test()
