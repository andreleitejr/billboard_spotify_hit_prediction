import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime

index_col = 'Song'
target = 'Peak Position'


def load_data(file_path: str) -> pd.DataFrame:
    """Loads a dataset from a CSV file using 'Id' as the index."""
    data = pd.read_csv(file_path, index_col=index_col)
    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data['performance'] = data['Peak Position'].apply(categorize_peak)
    data['energy_level'] = data['Energy'].apply(categorize_energy)

    current_year = datetime.now().year
    data['time_since_release'] = current_year - data['Release Year']
    data['log_streams'] = np.log1p(data['Streams'])
    data['log_daily_streams'] = np.log1p(data['Daily Streams'])
    data['log_weeks_on_chart'] = np.log1p(data['Weeks on Chart'])
    data['stream_growth_rate'] = data['Daily Streams'] / (data['Streams'] + 1e-6)
    data['abs_lyrics_sentiment'] = data['Lyrics Sentiment'].abs()

    return data


def categorize_peak(position):
    if 1 <= position <= 10:
        return 'Top'
    elif 11 <= position <= 30:
        return 'Good'
    elif 31 <= position <= 60:
        return 'Medium'
    else:
        return 'Bad'


def categorize_energy(level):
    if level >= 0.75:
        return 'Very High'
    elif level >= 0.5:
        return 'High'
    elif level >= 0.25:
        return 'Medium'
    else:
        return 'Low'


def columns_to_drop() -> list:
    """Returns a list of columns to remove."""
    return [target, 'Artist', ]


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=columns_to_drop()).copy()
    y = data[target].copy()
    y = y.max() - y + 1
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid


def preprocess_data(X_train: pd.DataFrame, X_valid: pd.DataFrame, preprocessor_path: str) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    """Applies numerical imputation, categorical encoding, and scaling."""
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)
    joblib.dump(preprocessor, preprocessor_path)

    return X_train_transformed, X_valid_transformed
