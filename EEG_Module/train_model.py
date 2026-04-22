#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced EEG Emotion Prediction Pipeline (Improved XGBoost)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

# ---------------------------------------------------
# 1 LOAD DATA
# ---------------------------------------------------

path = "/serverdata/ccshome/anjanasinha/NAS/DreamData/DSU/preprocessed_v2.csv"

print("Loading dataset...")

df = pd.read_csv(path)

print("Dataset shape:", df.shape)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ---------------------------------------------------
# 2 EMOTION DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Emotion", data=df)
plt.title("Emotion Distribution")
plt.savefig("emotion_distribution.png", dpi=300)
plt.show()

# ---------------------------------------------------
# 3 SPLIT FEATURES
# ---------------------------------------------------

X = df.drop("Emotion", axis=1)
y = df["Emotion"]

# ---------------------------------------------------
# 4 CLEAN DATA
# ---------------------------------------------------

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

mask = ~X.isna().any(axis=1)

X = X.loc[mask]
y = y.loc[mask]

print("Dataset after cleaning:", X.shape)

# ---------------------------------------------------
# 5 TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)

# ---------------------------------------------------
# 6 FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# 7 FEATURE SELECTION
# ---------------------------------------------------

print("Selecting important features...")

temp_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1
)

temp_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(
    temp_model,
    prefit=True,
    threshold=-np.inf,
    max_features=60
)

X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]

print("Selected features:", len(selected_features))

# ---------------------------------------------------
# 8 TRAIN IMPROVED XGBOOST
# ---------------------------------------------------

print("Training XGBoost...")

model = XGBClassifier(
    n_estimators=1200,
    max_depth=14,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    min_child_weight=2,
    reg_alpha=0.4,
    reg_lambda=1.6,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

# Custom class weighting
weights = compute_sample_weight(
    class_weight={0:1.4,1:1.0,2:1.0},
    y=y_train
)

model.fit(
    X_train_sel,
    y_train,
    sample_weight=weights,
    eval_set=[(X_test_sel, y_test)],
    verbose=50
)

# ---------------------------------------------------
# 9 PREDICTIONS
# ---------------------------------------------------

preds = model.predict(X_test_sel)

accuracy = accuracy_score(y_test, preds)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, preds))

# ---------------------------------------------------
# 10 CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 11 ROC CURVES
# ---------------------------------------------------

classes = np.unique(y)

y_test_bin = label_binarize(y_test, classes=classes)

y_pred_prob = model.predict_proba(X_test_sel)

plt.figure(figsize=(7,6))

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_pred_prob[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {classes[i]} AUC={roc_auc:.2f}")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")

plt.legend()

plt.savefig("roc_curves.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 12 FEATURE IMPORTANCE
# ---------------------------------------------------

importance = model.feature_importances_

indices = np.argsort(importance)[::-1][:20]

top_features = selected_features[indices]

plt.figure(figsize=(8,6))

sns.barplot(x=importance[indices], y=top_features)

plt.title("Top EEG Feature Importance")

plt.savefig("feature_importance.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 13 SHAP EXPLAINABILITY
# ---------------------------------------------------

print("Generating SHAP plots...")

try:
    explainer = shap.TreeExplainer(model.get_booster())
    shap_values = explainer.shap_values(X_test_sel[:1000])
    shap.summary_plot(shap_values, X_test_sel[:1000], show=False)
    plt.savefig("shap_summary.png", dpi=300)

except Exception as e:
    print("SHAP skipped due to compatibility issue:", e)

# ---------------------------------------------------
# 14 EEG BAND IMPORTANCE
# ---------------------------------------------------

bands = {
"Delta":[f for f in selected_features if "Delta" in f],
"Theta":[f for f in selected_features if "Theta" in f],
"Alpha":[f for f in selected_features if "Alpha" in f],
"Beta":[f for f in selected_features if "Beta" in f],
"Gamma":[f for f in selected_features if "Gamma" in f]
}

band_importance = {}

for band, feats in bands.items():
    idx = [i for i,f in enumerate(selected_features) if f in feats]
    band_importance[band] = importance[idx].sum() if idx else 0

plt.figure(figsize=(6,4))

sns.barplot(
    x=list(band_importance.keys()),
    y=list(band_importance.values())
)

plt.title("EEG Band Importance")

plt.savefig("band_importance.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 15 CROSS VALIDATION
# ---------------------------------------------------

print("Starting cross validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for fold,(train_idx,test_idx) in enumerate(cv.split(X,y)):

    print("Fold", fold+1)

    Xtr = X.iloc[train_idx]
    Xte = X.iloc[test_idx]

    ytr = y.iloc[train_idx]
    yte = y.iloc[test_idx]

    scaler_cv = StandardScaler()

    Xtr = scaler_cv.fit_transform(Xtr)
    Xte = scaler_cv.transform(Xte)

    temp_model.fit(Xtr, ytr)

    selector_cv = SelectFromModel(
        temp_model,
        prefit=True,
        threshold=-np.inf,
        max_features=60
    )

    Xtr = selector_cv.transform(Xtr)
    Xte = selector_cv.transform(Xte)

    weights_cv = compute_sample_weight(
        class_weight={0:1.4,1:1.0,2:1.0},
        y=ytr
    )

    model.fit(Xtr, ytr, sample_weight=weights_cv)

    score = model.score(Xte, yte)

    scores.append(score)

print("\nCross Validation Accuracy:", np.mean(scores))

# ---------------------------------------------------
# 16 t-SNE VISUALIZATION
# ---------------------------------------------------

print("Generating t-SNE plot...")

sample_idx = np.random.choice(len(X_train_sel), 1500, replace=False)

sample = X_train_sel[sample_idx]

labels_sample = y_train.iloc[sample_idx]

tsne = TSNE(n_components=2, random_state=42)

X_tsne = tsne.fit_transform(sample)

plt.figure(figsize=(7,6))

sns.scatterplot(
    x=X_tsne[:,0],
    y=X_tsne[:,1],
    hue=labels_sample,
    palette="Set1",
    s=40
)

plt.title("t-SNE Emotion Clustering")

plt.savefig("tsne_clusters.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 17 SAVE MODEL
# ---------------------------------------------------

joblib.dump(model,"eeg_xgboost_model.pkl")
joblib.dump(scaler,"eeg_scaler.pkl")
joblib.dump(selector,"feature_selector.pkl")
joblib.dump(selected_features,"selected_features.pkl")

print("\nModel and preprocessing objects saved successfully")