# Bank Marketing Classification - Machine Learning Assignment 2

## a. Problem Statement
Predict whether a client will subscribe to a term deposit (`y` = yes/no) based on demographic and marketing campaign data. This is a binary classification task.

## b. Dataset Description
- **Source**: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Training Data**: `bank-full.csv` – 45,211 instances, 16 features, 1 target (`y`)
- **Test Data Sample**: `bank.csv` – a subset of 4,521 rows (provided for demo upload in the Streamlit app)
- **Features**:
  - `age` (numeric)
  - `job` (categorical)
  - `marital` (categorical)
  - `education` (categorical)
  - `default` (categorical)
  - `balance` (numeric)
  - `housing` (categorical)
  - `loan` (categorical)
  - `contact` (categorical)
  - `day` (numeric)
  - `month` (categorical)
  - `duration` (numeric)
  - `campaign` (numeric)
  - `pdays` (numeric)
  - `previous` (numeric)
  - `poutcome` (categorical)
- **Target**: `y` – “yes” (client subscribed) / “no” (did not subscribe)

## c. Models Used

Six classification models were implemented and evaluated on the same dataset. The evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient) were computed on the test set.

### Comparison of Evaluation Metrics

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.891    | 0.925 | 0.665     | 0.451  | 0.537 | 0.461 |
| Decision Tree       | 0.876    | 0.842 | 0.602     | 0.427  | 0.500 | 0.411 |
| K-Nearest Neighbors | 0.885    | 0.898 | 0.638     | 0.439  | 0.520 | 0.438 |
| Naive Bayes         | 0.858    | 0.899 | 0.553     | 0.518  | 0.535 | 0.418 |
| Random Forest       | 0.901    | 0.940 | 0.703     | 0.470  | 0.563 | 0.495 |
| XGBoost             | 0.904    | 0.945 | 0.711     | 0.478  | 0.572 | 0.506 |

### Observations on Model Performance

| ML Model Name       | Observation |
|---------------------|-------------|
| Logistic Regression | Provides a strong linear baseline with good overall accuracy and AUC. However, recall is relatively low, indicating it misses a fair number of positive (subscribed) cases. |
| Decision Tree       | Easy to interpret but tends to overfit the training data. Performance is moderate; lower AUC and MCC compared to ensemble methods. |
| K-Nearest Neighbors | Sensitive to feature scaling and distance metrics. Achieves decent accuracy but recall remains low. Performance can vary with the choice of `k`. |
| Naive Bayes         | Assumes feature independence, which is not strictly true. Nevertheless, it yields a higher recall than logistic regression, capturing more positive instances at the cost of precision. |
| Random Forest       | Ensemble of decision trees reduces overfitting and improves generalization. It achieves the second‑best overall metrics, with a good balance between precision and recall. |
| XGBoost             | Gradient boosting delivers the highest accuracy, AUC, and MCC. It slightly outperforms Random Forest, making it the best model for this dataset. |

---

## Additional Information

### Repository Structure
