import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# ---------------- Config ----------------
CSV = "augmented_gaze_data_balanced.csv"
LABEL_COL = "Emotion"
FEATURES = ["avg_x", "avg_y", "std_x", "std_y", "avg_pupil", "saccade_speed"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
K = 3

OUTPUT_DIR = Path("Eye_Gaze_Module/Model_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIST_DIR = Path(".dist")
DIST_DIR.mkdir(exist_ok=True)

# ---------------- Load ----------------
df = pd.read_csv(CSV)
X = df[FEATURES].copy()
y = df[LABEL_COL].astype(str)

# ---------------- Encode labels ----------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

class_names = list(le.classes_)
display_names = [
    name.replace("_EMOTION_RATING", "").replace("_emotion_rating", "").lower()
    for name in class_names
]
print("Encoder classes :", class_names)
print("Display names   :", display_names)

# ---------------- Train / Test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_enc
)

# ---------------- Pipeline ----------------
rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1
    ))
])

param_grid = {
    "clf__max_depth": [8, 10],
    "clf__min_samples_leaf": [5, 10],
    "clf__max_features": ["sqrt"]
}

cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(rf_pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=1, verbose=2)

print("\nStarting GridSearchCV...")
gs.fit(X_train, y_train)

print(f"\nBest CV accuracy : {gs.best_score_:.4f}")
print("Best params      :", gs.best_params_)

best_rf = gs.best_estimator_

# ---------------- Evaluate ----------------
y_pred = best_rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {round(test_acc, 4)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=display_names, zero_division=0))

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=display_names, yticklabels=display_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RandomForest — Confusion Matrix (held-out test)")
plt.tight_layout()
cm_path = OUTPUT_DIR / "rf_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
display(Image(str(cm_path)))
print(f"Confusion matrix saved → {cm_path}")

# ---------------- Feature Importance ----------------
rf_model = best_rf.named_steps["clf"]
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(data=imp_df, x="importance", y="feature")
plt.title("RandomForest — Feature Importance")
plt.tight_layout()
fi_path = OUTPUT_DIR / "rf_feature_importance.png"
plt.savefig(fi_path, dpi=150)
plt.close()
display(Image(str(fi_path)))
print(f"Feature importance saved → {fi_path}")

# ---------------- Save Artifacts ----------------
joblib.dump(best_rf,  DIST_DIR / "rf_regularized_model.pkl")
joblib.dump(le,       DIST_DIR / "label_encoder.pkl")
joblib.dump(FEATURES, DIST_DIR / "feature_list.pkl")
print(f"\nAll artifacts saved in {DIST_DIR}/ ✓")
