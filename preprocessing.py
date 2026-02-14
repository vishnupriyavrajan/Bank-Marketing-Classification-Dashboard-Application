# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET = "y"

def preprocess_train_data(df: pd.DataFrame):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

    cat_cols = df.select_dtypes(include="object").columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    return X, y, encoders


def preprocess_test_data(df: pd.DataFrame, encoders):
    df = df.copy()
    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})
    y_test = df[TARGET]

    df.drop(TARGET, axis=1, inplace=True)

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    return df, y_test
