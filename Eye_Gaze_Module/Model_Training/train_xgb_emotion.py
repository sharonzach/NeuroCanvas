# ===============================
# Robust XGBoost Emotion Classifier
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- 1. Load dataset ---
input_path = r"E:\Drive E files\MAJOR PROJECT\eyegaze_emotion_project\nimhans\augmented_gaze_data.csv"
df = pd.read_csv(input_path)
print("✅ Dataset loaded. Shape:", df.shape)

# --- 2. Prepare features & target ---
feature_cols = ['avg_x', 'avg_y', 'std_x', 'std_y', 'avg_pupil', 'saccade_speed']
X = df[feature_cols].values
y = df['Emotion'].values

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 4. Initialize XGBoost classifier ---
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)

# --- 5. Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print("5-Fold CV Accuracy scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# --- 6. Train model ---
model.fit(X_train, y_train)

# --- 7. Evaluate on test set ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")

# --- Classification Report ---
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------------------------------------
# --- 8. Confusion Matrix (with cleaned emotion names) ------
# -----------------------------------------------------------

# Auto-clean labels (convert: 'anger_EMOTION_RATING' → 'Anger')
pretty_names = [label.split('_')[0].capitalize() for label in le.classes_]

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=pretty_names,
    yticklabels=pretty_names,
    cmap='Blues'
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- 9. Feature Importance ---
plt.figure(figsize=(10,6))
xgb.plot_importance(model, height=0.5)
plt.title("Feature Importance")
plt.show()

# --- 10. Save model ---
output_model_path = r"E:\Drive E files\MAJOR PROJECT\eyegaze_emotion_project\nimhans\xgb_saccade_model.pkl"
joblib.dump((model, le, feature_cols, scaler), output_model_path)
print(f"\n✅ Model saved at: {output_model_path}")
