# IT2011 - Letter Recognition using SVM
# Individual SVM part only

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Folder containing the prepared datasets
DATA_PATH = "data"

# Load scaled train and test data
scaled_train = pd.read_csv(os.path.join(DATA_PATH, "pipeline_scaled_train.csv"))
scaled_test = pd.read_csv(os.path.join(DATA_PATH, "pipeline_scaled_test.csv"))

# Load PCA train and test data
pca_train = pd.read_csv(os.path.join(DATA_PATH, "pipeline_pca_train.csv"))
pca_test = pd.read_csv(os.path.join(DATA_PATH, "pipeline_pca_test.csv"))

# Separate input features (X) and target labels (y)
X_train = scaled_train.drop(columns=["letter"])
y_train = scaled_train["letter"]
X_test = scaled_test.drop(columns=["letter"])
y_test = scaled_test["letter"]

X_train_pca = pca_train.drop(columns=["letter"])
y_train_pca = pca_train["letter"]
X_test_pca = pca_test.drop(columns=["letter"])
y_test_pca = pca_test["letter"]

# Train Linear SVM
linear_model = SVC(kernel="linear", C=1)
linear_model.fit(X_train, y_train)
linear_acc = accuracy_score(y_test, linear_model.predict(X_test))

# Train RBF SVM
rbf_model = SVC(kernel="rbf", C=1, gamma="scale")
rbf_model.fit(X_train, y_train)
rbf_acc = accuracy_score(y_test, rbf_model.predict(X_test))

# Train Polynomial SVM
poly_model = SVC(kernel="poly", C=1, degree=3, gamma="scale")
poly_model.fit(X_train, y_train)
poly_acc = accuracy_score(y_test, poly_model.predict(X_test))

# Compare RBF SVM with PCA data
pca_model = SVC(kernel="rbf", C=1, gamma="scale")
pca_model.fit(X_train_pca, y_train_pca)
pca_acc = accuracy_score(y_test_pca, pca_model.predict(X_test_pca))

print("Linear SVM Accuracy:", round(linear_acc * 100, 2), "%")
print("RBF SVM Accuracy:", round(rbf_acc * 100, 2), "%")
print("Polynomial SVM Accuracy:", round(poly_acc * 100, 2), "%")
print("RBF + PCA Accuracy:", round(pca_acc * 100, 2), "%")

# Test a small set of RBF hyperparameters using 3-fold cross-validation
param_grid = {
    "kernel": ["rbf"],
    "C": [1, 10],
    "gamma": ["scale", 0.1]
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

# Evaluate the best tuned SVM on the test data
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)
best_acc = accuracy_score(y_test, best_pred)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", round(grid.best_score_ * 100, 2), "%")
print("Final Test Accuracy:", round(best_acc * 100, 2), "%")

# Print precision, recall and F1-score for A-Z
letters = [chr(65 + i) for i in range(26)]
print(classification_report(y_test, best_pred, target_names=letters))

# Display confusion matrix
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay(cm, display_labels=letters).plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)
plt.title("Best SVM - Confusion Matrix")
plt.tight_layout()
plt.show()

# Compare the main SVM results
results = pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM", "Polynomial SVM", "RBF + PCA", "Tuned RBF SVM"],
    "Accuracy (%)": [
        linear_acc * 100,
        rbf_acc * 100,
        poly_acc * 100,
        pca_acc * 100,
        best_acc * 100
    ]
})
results["Accuracy (%)"] = results["Accuracy (%)"].round(2)
print(results)

# Draw a simple accuracy comparison chart
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["Accuracy (%)"])
plt.ylabel("Accuracy (%)")
plt.title("SVM Model Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Save the best tuned model
joblib.dump(best_model, "best_svm_model.pkl")
print("Model saved as best_svm_model.pkl")
