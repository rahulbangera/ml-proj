

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as stats


print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
df = pd.read_csv(path + "/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n" + "=" * 60)
print("2. DATA PREPROCESSING")
print("=" * 60)

columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

columns_to_drop.append('PercentSalaryHike')

columns_to_drop.extend(['DailyRate', 'HourlyRate', 'MonthlyRate'])

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print(f"Dropped columns: {columns_to_drop}")
print(f"Remaining columns: {list(df.columns)}")

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Encoding categorical columns: {list(categorical_cols)}")
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"\nDataset shape after preprocessing: {df.shape}")
print(f"\nTarget distribution (PerformanceRating):")
print(df['PerformanceRating'].value_counts())

print("\n" + "=" * 60)
print("3. TRAIN/TEST SPLIT (before resampling)")
print("=" * 60)

X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"\nTraining target distribution:")
print(y_train.value_counts())
print(f"\nTest target distribution:")
print(y_test.value_counts())

print("\n" + "=" * 60)
print("4. RESAMPLING TRAINING DATA ONLY")
print("=" * 60)

train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['PerformanceRating'] = y_train.reset_index(drop=True)

df_majority = train_df[train_df['PerformanceRating'] == 3]
df_minority = train_df[train_df['PerformanceRating'] == 4]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_resampled = pd.concat([df_majority, df_minority_upsampled])
df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_resampled = df_resampled.drop('PerformanceRating', axis=1)
y_train_resampled = df_resampled['PerformanceRating']

print(f"Resampled training target distribution:")
print(y_train_resampled.value_counts())

print("\n" + "=" * 60)
print("5. OUTLIER REMOVAL")
print("=" * 60)

z = np.abs(stats.zscore(X_train_resampled))
mask = (z < 3).all(axis=1)
X_train_clean = X_train_resampled[mask]
y_train_clean = y_train_resampled[mask]

print(f"Training samples after outlier removal: {len(X_train_clean)}")
print(f"Training target distribution after outlier removal:")
print(y_train_clean.value_counts())

print("\n" + "=" * 60)
print("6. FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler fitted on training data only.")
print(f"Training features shape: {X_train_scaled.shape}")
print(f"Test features shape: {X_test_scaled.shape}")

print("\n" + "=" * 60)
print("7. MODEL TRAINING AND EVALUATION")
print("=" * 60)

results = {}

# --- 7a. Support Vector Machine ---
print("\n--- Support Vector Machine (RBF kernel) ---")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train_clean)
y_pred_svm = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_pred_svm)
results['SVM'] = svm_acc
print(f"SVM Accuracy: {svm_acc:.2%}")
print(f"\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# --- 7b. Logistic Regression ---
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_clean)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred_lr)
results['Logistic Regression'] = lr_acc
print(f"Logistic Regression Accuracy: {lr_acc:.2%}")
print(f"\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# --- 7c. Gradient Boosting Classifier ---
print("\n--- Gradient Boosting Classifier ---")
gbc_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
gbc_model.fit(X_train_scaled, y_train_clean)
y_pred_gbc = gbc_model.predict(X_test_scaled)
gbc_acc = accuracy_score(y_test, y_pred_gbc)
results['Gradient Boosting'] = gbc_acc
print(f"Gradient Boosting Accuracy: {gbc_acc:.2%}")
print(f"\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gbc))

# --- 7d. K-Nearest Neighbors ---
print("\n--- K-Nearest Neighbors (k=5) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_clean)
y_pred_knn = knn_model.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, y_pred_knn)
results['KNN'] = knn_acc
print(f"KNN Accuracy: {knn_acc:.2%}")
print(f"\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# --- 7e. Random Forest ---
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_clean)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, y_pred_rf)
results['Random Forest'] = rf_acc
print(f"Random Forest Accuracy: {rf_acc:.2%}")
print(f"\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# --- 7f. Decision Tree ---
print("\n--- Decision Tree Classifier ---")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train_clean)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_acc = accuracy_score(y_test, y_pred_dt)
results['Decision Tree'] = dt_acc
print(f"Decision Tree Accuracy: {dt_acc:.2%}")
print(f"\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# --- 7g. Gaussian Naive Bayes ---
print("\n--- Gaussian Naive Bayes ---")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train_clean)
y_pred_nb = nb_model.predict(X_test_scaled)
nb_acc = accuracy_score(y_test, y_pred_nb)
results['Naive Bayes'] = nb_acc
print(f"Naive Bayes Accuracy: {nb_acc:.2%}")
print(f"\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

print("\n" + "=" * 60)
print("8. RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25} {'Accuracy':<10}")
print("-" * 35)
for model_name, acc in results.items():
    print(f"{model_name:<25} {acc:.2%}")

print("\nNote: These realistic accuracies (typically 70-90%) are expected")
print("because the leaky feature 'PercentSalaryHike' has been removed,")
print("and the data pipeline now properly prevents train/test leakage.")
