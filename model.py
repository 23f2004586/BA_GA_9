import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

#----------------STEP 1: Load Data ----------------
train = pd.read_csv("train_LR.csv")
test = pd.read_csv("test_LR.csv")

# ---------------- STEP 2: Create X and y ----------------
X_train = train.drop("Accept", axis=1)
y_train = train["Accept"]

X_test = test.drop("Accept", axis=1)
y_test = test["Accept"]

# ---------------- STEP 3: Proportions ----------------
train_not_accept = (y_train.value_counts()[0] / len(y_train)) * 100
train_accept = (y_train.value_counts()[1] / len(y_train)) * 100
test_not_accept = (y_test.value_counts()[0] / len(y_test)) * 100
test_accept = (y_test.value_counts()[1] / len(y_test)) * 100

# ---------------- STEP 4: Logistic Regression ----------------
model = LogisticRegression(random_state=17)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#---------------- STEP 5: Train Metrics ----------------
train_acc = accuracy_score(y_train, y_pred_train) * 100
train_precision_not_accept = precision_score(y_train, y_pred_train, pos_label=0) * 100
train_recall_accept = recall_score(y_train, y_pred_train, pos_label=1) * 100

# ---------------- STEP 6: Test Metrics ----------------
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
test_acc = accuracy_score(y_test, y_pred_test) * 100
test_precision_not_accept = precision_score(y_test, y_pred_test, pos_label=0) * 100
test_recall_accept = recall_score(y_test, y_pred_test, pos_label=1) * 100

# Precision comparison (accept class)
train_precision_accept = precision_score(y_train, y_pred_train, pos_label=1) * 100
test_precision_accept = precision_score(y_test, y_pred_test, pos_label=1) * 100
precision_change = "Increased" if test_precision_accept > train_precision_accept else "Decreased"

# ---------------- STEP 7: Print All Answers ----------------
print("Q1:", train.shape[0])
print("Q2:", test.shape[0])
print("Q3:", round(train_not_accept, 2))
print("Q4:", round(train_accept, 2))
print("Q5:", round(test_not_accept, 2))
print("Q6:", round(test_accept, 2))
print("Q7:", round(train_acc, 3))
print("Q8:", round(train_precision_not_accept, 3))
print("Q9:", round(train_recall_accept, 2))
print("Q10:", fp)
print("Q11:", fn)
print("Q12:", round(test_acc, 2))
print("Q13:", round(test_precision_not_accept, 2))
print("Q14:", round(test_recall_accept, 2))
print("Q15:", precision_change)
