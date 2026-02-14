from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocess_train_data, preprocess_test_data
from metrics import evaluate_model
from data_loader import load_data

TRAIN_PATH = "bank-full.csv"

def run_decision_tree(test_df):
    train_df = load_data(TRAIN_PATH)
    X_train, y_train, encoders = preprocess_train_data(train_df)

    model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    X_test, y_test = preprocess_test_data(test_df, encoders)

    return evaluate_model(model, X_test, y_test)
