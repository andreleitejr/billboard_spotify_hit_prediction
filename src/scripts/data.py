import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.utils.config import PREPROCESSOR_PATH

target = 'Peak Position'


def load_data(file_path: str) -> pd.DataFrame:
    """Loads a dataset from a CSV file using 'Id' as the index."""
    data = pd.read_csv(file_path, delimiter=',', encoding='ascii')
    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Apply custom feature engineering to enhance the dataset."""
    data['Hit'] = data['Peak Position'].apply(lambda x: 1 if x <= 10 else 0)

    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/validation sets and encode 'Genre'."""
    X = data.drop(['Peak Position', 'Hit', 'Song', 'Artist'], axis=1)
    X = pd.get_dummies(X, columns=['Genre'], drop_first=True)
    y = data['Hit']
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def preprocess_data(X_train) -> pd.DataFrame:
    """Applies numerical imputation, categorical encoding, and scaling."""
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])

    joblib.dump(X_train.columns.tolist(), PREPROCESSOR_PATH)
    return X_train
