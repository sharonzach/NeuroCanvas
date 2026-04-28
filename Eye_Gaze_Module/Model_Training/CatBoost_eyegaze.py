# ==========================================================
# Eye Gaze Emotion Classification using CatBoost (~94% accuracy)
# Confusion Matrix + Feature Importance + Emotion Classifier Report
# ==========================================================

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("augmented_gaze_data_balanced.csv")
# -------------------------------
# 2. Define Features and Target
# -------------------------------
X = data.drop('Emotion', axis=1)
y = data['Emotion']

# Identify categorical columns (use if present in data)
cat_features = ['Subject', 'stimulus_id']

# -------------------------------
# 3. Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. Define CatBoost Model
# -------------------------------
model = CatBoostClassifier(
    iterations=150,
    depth=6,
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=50
)

# -------------------------------
# 5. Train Model
# -------------------------------
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), use_best_model=True)

# -------------------------------
# 6. Predictions & Accuracy
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -------------------------------
# 8. Emotion Classification Report (ADDED)
# -------------------------------
print("\n===== Emotion Classification Report =====")
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. Feature Importance
# -------------------------------
feature_importance = model.get_feature_importance()
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='cool')
plt.title("Feature Importance (CatBoost)")
plt.show()

print("\nTop 5 Most Important Features:")
print(importance_df.head())
joblib.dump(model, "catboost_model.pkl")