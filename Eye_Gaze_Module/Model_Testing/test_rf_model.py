"""
Test the saved Random Forest gaze-emotion model on new data.
"""

import argparse
import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ---------------- Config ----------------
MODEL_PATH    = ".dist/rf_regularized_model.pkl"
ENCODER_PATH  = ".dist/label_encoder.pkl"
FEATURES_PATH = ".dist/feature_list.pkl"
TEST_CSV      = "augmented_gaze_data_balanced.csv"   # replace with your test CSV
LABEL_COL     = "Emotion"

OUTPUT_DIR = Path("Eye_Gaze_Module/Model_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load artifacts ----------------
print("Loading model artifacts...")
model    = joblib.load(MODEL_PATH)
le       = joblib.load(ENCODER_PATH)
features = joblib.load(FEATURES_PATH)

class_names   = list(le.classes_)
display_names = [
    n.replace("_EMOTION_RATING", "").replace("_emotion_rating", "").lower()
    for n in class_names
]
print("Classes:", display_names)

# ---------------- Load test data ----------------
df     = pd.read_csv(TEST_CSV)
X_test = df[features].copy()

# ---------------- Predict ----------------
y_pred_enc    = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred_enc)

df["Predicted_Emotion"] = y_pred_labels
out_csv = OUTPUT_DIR / "test_predictions.csv"
df.to_csv(out_csv, index=False)

print(f"\nPredictions saved → {out_csv}")
print("\nPrediction counts:")
print(df["Predicted_Emotion"].value_counts().to_string())

# ---------------- Evaluate if ground truth exists ----------------
if LABEL_COL in df.columns:
    y_true     = df[LABEL_COL].astype(str)
    y_true_enc = le.transform(y_true)

    acc = accuracy_score(y_true_enc, y_pred_enc)
    print(f"\nTest Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_enc, y_pred_enc,
                                target_names=display_names,
                                zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=display_names, yticklabels=display_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("RF Model — Test Confusion Matrix")
    plt.tight_layout()

    cm_path = OUTPUT_DIR / "rf_test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    display(Image(str(cm_path)))
    print(f"\nTest confusion matrix saved → {cm_path}")

else:
    print(f"\nNo '{LABEL_COL}' column found — skipping evaluation.")
