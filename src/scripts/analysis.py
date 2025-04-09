import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def numeric_histogram(data: pd.DataFrame) -> None:
    """Displays histograms for all numeric features in the dataset."""
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_features = numeric_data.columns

    plt.figure(figsize=(15, 10))

    for i, col in enumerate(numeric_features, 1):
        plt.subplot(3, 4, i)
        sns.histplot(numeric_data[col], kde=True, bins=20)
        plt.title(col)

    plt.tight_layout()
    plt.show()


def numeric_pair_plot(data: pd.DataFrame) -> None:
    """Generates pairwise scatter plots for numeric features."""
    numeric_data = data.select_dtypes(include=[np.number])

    sns.pairplot(numeric_data, corner=True, plot_kws={'alpha': 0.5, 's': 25})

    plt.show()


def numeric_correlation_heatmap(data: pd.DataFrame) -> None:
    """Shows a heatmap of correlations between numeric features."""
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] >= 4:
        plt.figure(figsize=(10, 8))
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Numeric Features')
        plt.show()
    else:
        print('Not enough numeric columns for a meaningful correlation heatmap.')

def categorical_count_plot(data: pd.DataFrame) -> None:
    """Displays count plots for all categorical features in the dataset, except Peak Position, Song and Artist."""
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=data, x=col, order=data[col].value_counts().index)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(y_valid, predictions, labels) -> None:
    """Plots a confusion matrix comparing true and predicted labels."""
    cm = confusion_matrix(y_valid, predictions)

    plt.figure(figsize=(6, 4))

    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues', cbar=False,
                xticklabels=labels,
                yticklabels=labels)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
