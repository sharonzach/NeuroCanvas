# EEG Emotion Prediction Module

## Overview
This module predicts emotional states from EEG features using an advanced XGBoost pipeline.

## Pipeline Steps
1. Data Loading
2. Preprocessing and Cleaning
3. Label Encoding
4. SMOTE Balancing
5. Feature Scaling
6. Feature Selection
7. Model Training (XGBoost)
8. Evaluation and Visualization

## Files

- preprocessing.py → Data cleaning + SMOTE
- train_model.py → Model training pipeline
- test_model.py → Model evaluation

## Outputs

### Confusion Matrix
![Confusion](Output/confusion_matrix.png)

### ROC Curves
![ROC](Output/roc_curves.png)

### Feature Importance
![Importance](Output/feature_importance.png)

### t-SNE Clustering
![TSNE](Output/tsne_clusters.png)

## Model Details
- Algorithm: XGBoost
- Multi-class classification
- Feature selection using importance ranking
- Class balancing using SMOTE

## Note
Raw EEG dataset is not included due to NIMHANS data restrictions.
