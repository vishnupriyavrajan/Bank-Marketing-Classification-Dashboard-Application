from xgboost import XGBClassifier
from preprocessing import preprocess_train_data, preprocess_test_data
from metrics import evaluate_model
from data_loader import load_data

TRAIN_PATH = "bank-full.csv"

def run_xgboost(test_df):
    train_df = load_data(TRAIN_PATH)
    X_train, y_train, encoders = preprocess_train_data(train_df)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    X_test, y_test = preprocess_test_data(test_df, encoders)

    return evaluate_model(model, X_test, y_test)
