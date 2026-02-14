import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    classification_report
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    # ---- Metrics Table ----
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    metrics_df = pd.DataFrame([metrics])

    # ---- Classification Report ----
    report_dict = classification_report(
        y_test, y_pred, output_dict=True
    )

    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Class"})
    )

    # Round for clean display
    report_df = report_df.round(3)

    return metrics_df, report_df, report_dict
