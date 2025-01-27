import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the Titanic dataset
print("Loading the dataset...")
df = pd.read_csv('gender_submission.csv')

# Check dataset structure
print("Columns in the dataset:", df.columns)
print(df.head())
print(df.info())

# Preparing data for ML model
print("Preparing data for ML model...")

# Create new synthetic feature for demonstration
df['is_prime'] = df['PassengerId'].apply(
    lambda x: all(x % i != 0 for i in range(2, int(np.sqrt(x)) + 1)) and x > 1
).astype(int)  # Marking prime IDs for diversity in data.

df['id_modulus'] = df['PassengerId'] % 10  # Adding a categorical-like feature.
df['normalized_passenger_id'] = (df['PassengerId'] - df['PassengerId'].mean()) / df['PassengerId'].std()

# Select features for the ML model
features = ['is_prime', 'id_modulus', 'normalized_passenger_id']
X = df[features]
y = df['Survived']

# Split data into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Support Vector Machine (SVM) model
print("Training Support Vector Machine (SVM) model...")
ml_model_svm = SVC(kernel='linear', random_state=42)
ml_model_svm.fit(X_train, y_train)

# Train a K-Nearest Neighbors (KNN) model
print("Training K-Nearest Neighbors (KNN) model...")
ml_model_knn = KNeighborsClassifier(n_neighbors=3)
ml_model_knn.fit(X_train, y_train)

# Predict using both ML models
print("Making predictions...")
y_pred_svm = ml_model_svm.predict(X_test)
y_pred_knn = ml_model_knn.predict(X_test)

# Evaluate both models' performance
print("Evaluating classifiers...")

# SVM evaluation
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)

# KNN evaluation
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_conf_matrix = confusion_matrix(y_test, y_pred_knn)

# Adding Confusion Matrix-derived metrics
def calculate_metrics(conf_matrix):
    TP = conf_matrix[1, 1]  # True Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

# Calculate additional metrics
svm_precision, svm_recall, svm_f1 = calculate_metrics(svm_conf_matrix)
knn_precision, knn_recall, knn_f1 = calculate_metrics(knn_conf_matrix)

# Display results
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print(f"SVM Precision: {svm_precision:.2f}, Recall: {svm_recall:.2f}, F1-Score: {svm_f1:.2f}")
print("Confusion Matrix for SVM Model:\n", svm_conf_matrix)

print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
print(f"KNN Precision: {knn_precision:.2f}, Recall: {knn_recall:.2f}, F1-Score: {knn_f1:.2f}")
print("Confusion Matrix for KNN Model:\n", knn_conf_matrix)

# Plot accuracies of both models
print("Creating accuracy graph...")

categories = ['SVM', 'KNN']
accuracy_values = [svm_accuracy, knn_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(categories, accuracy_values, color=['skyblue', 'lightgreen'], edgecolor='black')
plt.title('Comparison of Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for i, v in enumerate(accuracy_values):
    plt.text(i, v + 0.02, f'{v * 100:.2f}%', ha='center', va='bottom')

plt.show()
