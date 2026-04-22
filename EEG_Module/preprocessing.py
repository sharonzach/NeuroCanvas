#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG Emotion Prediction - Preprocessing with SMOTE
"""

import pandas as pd
import numpy as np
import subprocess
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------
# 1 FILE PATHS
# ---------------------------------------------------

eeg_path = "EEG PATH"
label_path = "LABEL PATH"

print("Loading datasets...")

# ---------------------------------------------------
# 2 LOAD EEG DATASET
# ---------------------------------------------------

print("Loading EEG dataset...")

cmd = f"cat {eeg_path}"

process = subprocess.Popen(
    cmd,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

output, error = process.communicate()

if process.returncode != 0:
    print("Error reading EEG file:")
    print(error.decode())
    raise SystemExit()

eeg = pd.read_csv(StringIO(output.decode()), low_memory=False)

print("EEG shape:", eeg.shape)

# ---------------------------------------------------
# 3 LOAD LABEL DATASET
# ---------------------------------------------------

labels = pd.read_csv(label_path)

print("Label shape:", labels.shape)

# ---------------------------------------------------
# 4 EXTRACT SUBJECT + DREAM ID FROM EEG
# ---------------------------------------------------

print("Extracting subject and dream IDs...")

eeg["Subj"] = eeg["SubjID"].astype(str).str.extract(r'([A-Z]{4})')

eeg["DreamID"] = eeg["SubjID"].astype(str).str.extract(r'D(\d+)')

eeg["DreamID"] = eeg["DreamID"].astype(float).astype("Int64")

# ---------------------------------------------------
# 5 PREPARE LABEL TABLE
# ---------------------------------------------------

labels = labels[["Subj", "SubjID", "Emotion"]]

labels = labels.rename(columns={"SubjID": "DreamID"})

labels["DreamID"] = labels["DreamID"].astype(int)

print("Label preview:")
print(labels.head())

# ---------------------------------------------------
# 6 MERGE EEG + LABELS
# ---------------------------------------------------

print("Merging EEG with emotion labels...")

merged = pd.merge(
    eeg,
    labels,
    on=["Subj", "DreamID"],
    how="left"
)

print("Merged shape:", merged.shape)

print("\nEmotion distribution before cleaning:")
print(merged["Emotion"].value_counts(dropna=False))

# ---------------------------------------------------
# 7 REMOVE ROWS WITHOUT LABELS
# ---------------------------------------------------

merged = merged.dropna(subset=["Emotion"])

print("Dataset after removing unlabeled rows:", merged.shape)

# ---------------------------------------------------
# 8 REMOVE METADATA COLUMNS
# ---------------------------------------------------

remove_columns = [
    "Subj",
    "SubjID",
    "DreamID",
    "Channel",
    "Condn",
    "Stage",
    "Subepochno"
]

existing_columns = [c for c in remove_columns if c in merged.columns]

X = merged.drop(columns=existing_columns + ["Emotion"])

y = merged["Emotion"]

print("Feature matrix shape:", X.shape)

# ---------------------------------------------------
# 9 CLEAN DATA (NaN / inf)
# ---------------------------------------------------

print("Cleaning dataset...")

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

mask = ~X.isna().any(axis=1)

X = X.loc[mask]

y = y.loc[mask]

print("Dataset after cleaning:", X.shape)

# ---------------------------------------------------
# 10 ENCODE EMOTION LABELS
# ---------------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("Emotion classes:", encoder.classes_)

print("Label mapping:")

print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# ---------------------------------------------------
# 11 APPLY SMOTE BALANCING
# ---------------------------------------------------

print("\nApplying SMOTE balancing...")

smote = SMOTE(random_state=42)

X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

print("Balanced dataset shape:", X_balanced.shape)

print("\nBalanced emotion distribution:")

print(pd.Series(y_balanced).value_counts())

# ---------------------------------------------------
# 12 CREATE FINAL DATASET
# ---------------------------------------------------

processed = pd.DataFrame(X_balanced, columns=X.columns)

processed["Emotion"] = y_balanced

# Remove accidental index column

if "Unnamed: 0" in processed.columns:
    processed = processed.drop(columns=["Unnamed: 0"])

# ---------------------------------------------------
# 13 SAVE DATASET
# ---------------------------------------------------

output_path = "preprocessed_v2.csv"

processed.to_csv(output_path, index=False)

print("\nBalanced dataset saved to:")

print(output_path)

# ---------------------------------------------------
# 14 FINAL CHECK
# ---------------------------------------------------

print("\nFinal dataset info")

print(processed.shape)

print("\nFinal emotion distribution:")

print(processed["Emotion"].value_counts())

print("\nSample rows:")

print(processed.head())
