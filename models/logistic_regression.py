from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_train_data, preprocess_test_data
from metrics import evaluate_model
from data_loader import load_data

TRAIN_PATH = "bank-full.csv"

def run_logistic_regression(test_df):
    # -------- TRAIN --------
    train_df = load_data(TRAIN_PATH)
    X_train, y_train, encoders = preprocess_train_data(train_df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # -------- TEST --------
    X_test, y_test = preprocess_test_data(test_df, encoders)

    return evaluate_model(model, X_test, y_test)
