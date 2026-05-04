# =======================
# 1. Import Libraries
# =======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =======================
# 2. Load Dataset
# =======================
# Replace with your actual CSV path
df = pd.read_csv("augmented_gaze_data_balanced.csv")

print("Shape:", df.shape)
print(df.head())

# =======================
# 3. Encode Labels
# =======================
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Emotion'] = le.fit_transform(df['Emotion'])

# =======================
# 4. Feature & Target Split
# =======================
X = df[['avg_x', 'avg_y', 'std_x', 'std_y', 'avg_pupil', 'saccade_speed']]
y = df['Emotion']
groups = df['Subject']  # for subject-wise cross-validation (optional)

# =======================
# 5. Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# 6. Define Models
# =======================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

svm = SVC(kernel='rbf', C=2, gamma='scale', probability=True, random_state=42)

# =======================
# 7. Ensemble Model
# =======================
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('svm', svm)],
    voting='soft'
)

# =======================
# 8. Train Model
# =======================
print("\nTraining Ensemble Model...")
voting_clf.fit(X_train, y_train)

# =======================
# 9. Evaluation
# =======================
y_pred = voting_clf.predict(X_test)

print("\n✅ Training Complete!")
print("\nTrain Accuracy:", voting_clf.score(X_train, y_train))
print("Test Accuracy:", voting_clf.score(X_test, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# =======================
# 10. Cross-Validation
# =======================
print("\nPerforming 5-Fold Cross Validation...")
cv_scores = cross_val_score(voting_clf, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# =======================
# 11. Subject-Wise Validation (Optional)
# =======================
# If your data has multiple subjects, use GroupKFold to ensure
# train and test sets don't mix samples from the same subject.
print("\nPerforming Subject-wise Cross-Validation (if applicable)...")

group_kfold = GroupKFold(n_splits=5)
subject_scores = cross_val_score(voting_clf, X, y, groups=groups, cv=group_kfold)
print("Subject-wise CV Scores:", subject_scores)
print("Mean Subject-wise Accuracy:", np.mean(subject_scores))
