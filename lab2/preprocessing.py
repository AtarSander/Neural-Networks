from scipy import stats
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
import torch


def classify(x):
    if x <= 100000:
        return 0
    if x <= 350000:
        return 1
    return 2


def unique(df, columns):
    return {column: df[column].nunique() for column in columns}


def missing(df, columns):
    return {
        column: df[column].isnull().sum()
        for column in columns
        if df[column].isnull().sum() > 0
    }


def plot_bar(values, title, size=(10, 4)):
    plt.figure(figsize=size)
    plt.bar(*zip(*values.items()))
    plt.title(title)
    plt.show()


def plot_boxplot(df, numeric_columns):
    size_x = len(numeric_columns) // 2
    size_y = len(numeric_columns) // size_x
    fig, axes = plt.subplots(size_x, size_y, figsize=(15, 25))
    for col, ax in zip(numeric_columns[:-1], axes.flatten()):
        ax.boxplot(df[col], patch_artist=True)
        ax.set_title(col)


def check_outliers(df, numeric_columns):
    outliers = {}
    outliers_counts = {}
    for col in numeric_columns[:-1]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_counts[col] = len(outliers[col])
    return outliers_counts, outliers


def plot_histograms(df, numeric_columns):
    size_x = len(numeric_columns) // 2
    size_y = len(numeric_columns) // size_x
    fig, axes = plt.subplots(size_x, size_y, figsize=(15, 25))
    for col, ax in zip(numeric_columns[:-1], axes.flatten()):
        df[col].plot(kind="hist", bins=20, ax=ax)
        ax.set_title(col)


def shapiro_wilk(df, numeric_columns):
    normally_distributed = {}
    positive = "normally distributed"
    negative = "not normally distributed"
    for col in numeric_columns[:-1]:
        shapiro_wilk_result = stats.shapiro(df[col])
        normally_distributed[col] = (
            positive if shapiro_wilk_result.pvalue >= 0.05 else negative
        )
    return normally_distributed


def calculate_class_weights(labels):
    numpy_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    return torch.from_numpy(numpy_weights).float()


def preprocess(df, numeric_columns, ordinal_columns, nominal_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_columns),
            ("numeric_scaler", MinMaxScaler(), numeric_columns),
            ("ordinal", OrdinalEncoder(), ordinal_columns),
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
        ]
    )
    pipeline.fit(df)
    return pipeline.transform(df)


# TODO check if ordinal encoding does ordering right
