# Bank Marketing Classification - Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to build and evaluate multiple classification models on the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit (y = yes / no). The task involves preprocessing the dataset, applying multiple machine learning algorithms, and comparing their performance using standard evaluation metrics.

Predict whether a client will subscribe to a term deposit (`y` = yes/no) based on demographic and marketing campaign data. This is a binary classification task.

## b. Dataset Description
- **Source**: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- The data is related with direct marketing campaigns of a Portuguese banking institution. 
  The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
  in order to access if the product (bank term deposit) would be (or not) subscribed. 

  There are two datasets: 
- **Training Data**: `bank-full.csv` – 45,211 instances, 16 features, 1 target (`y`)
- **Test Data Sample**: `bank.csv` – a subset of 4,521 rows (provided for demo upload in the Streamlit app)
  
- **Features**:
  
   Input variables:
   # bank client data:
   - age (numeric)
   - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   - education (categorical: "unknown","secondary","primary","tertiary")
   - default: has credit in default? (binary: "yes","no")
   - balance: average yearly balance, in euros (numeric) 
   - housing: has housing loan? (binary: "yes","no")
   - loan: has personal loan? (binary: "yes","no")
   related with the last contact of the current campaign:
   - contact: contact communication type (categorical: "unknown","telephone","cellular") 
   - day: last contact day of the month (numeric)
   - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
   - duration: last contact duration, in seconds (numeric)
   other attributes:
   - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
   - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means     client was not previously contacted)
   - previous: number of contacts performed before this campaign and for this client (numeric)
   - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
   - y - has the client subscribed a term deposit? (binary: "yes","no")

- Missing Attribute Values: None

- **Target**: `y` – “yes” (client subscribed) / “no” (did not subscribe)


## c. Models Used

Six classification models were implemented and evaluated on the same dataset. The evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient) were computed on the test set.

### Comparison of Evaluation Metrics

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.8894   | 0.8376| 0.56      | 0.1881 |0.2816 | 0.2795|
| Decision Tree       | 0.9073   |0.8735 | 0.6536    | 0.4165 | 0.5088| 0.4747|
| K-Nearest Neighbors | 0.9051   | 0.9227| 0.6533    | 0.3762 | 0.4775| 0.4493|
| Naive Bayes         | 0.8345   | 0.8026| 0.3319    | 0.4299 | 0.3746| 0.2842|
| Random Forest       | 0.91     | 0.9222| 0.7879    | 0.2994 | 0.4339| 0.4508|
| XGBoost             | 0.9281   | 0.9552| 0.7487    | 0.5662 | 0.6448| 0.613 |

### Observations on Model Performance

| ML Model Name       | Observation |
|---------------------|-------------|
| Logistic Regression | Achieved high accuracy but very low recall, indicating strong bias toward the majority class. The model struggles to correctly identify customers who subscribe, making it suitable only as a baseline model. |
| Decision Tree       | Demonstrated improved recall and F1-score by capturing non-linear relationships in the data. Model performance is balanced, but requires depth control to avoid overfitting. |
| K-Nearest Neighbors | Strong AUC but moderate recall, sensitive to feature scaling and data distribution. High AUC indicating good class separation.Distance-based learning works well but is sensitive to data distribution |
| Naive Bayes         | Assumes feature independence, which is not strictly true. Nevertheless, it yields a higher recall than logistic regression, capturing more positive instances at the cost of precision. |
| Random Forest       | Achieved high precision and accuracy by making conservative predictions. However, lower recall suggests the model prioritizes reducing false positives over detecting all subscribers. |
| XGBoost             | Outperformed all other models with the highest AUC, F1-score, and MCC. Effectively handled class imbalance and complex feature interactions, making it the most suitable model for deployment.|

---

## Additional Information

### Repository Structure
