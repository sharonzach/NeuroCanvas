import numpy as np

test_samples = np.array([
    # SAD (moderate movement, moderate pupil, medium speed)
    [0.37, 0.28, 0.17, 0.20, 3.25, 3.50],

    # HAPPY (lower std, smoother movement, moderate speed)
    [0.51, 0.49, 0.05, 0.04, 3.10, 3.20],

    # ANGER (high std + high speed)
    [0.42, 0.30, 0.21, 0.16, 3.20, 5.00],

    # FEAR (high speed + high pupil + scattered gaze)
    [0.60, 0.26, 0.18, 0.15, 3.70, 5.80],

    # DISGUST (lower avg_y + moderate std + mid speed)
    [0.56, 0.12, 0.03, 0.05, 3.45, 2.00]
])
import joblib

# Load model
model, le, feature_cols, scaler = joblib.load(
    r"xgb_saccade_model.pkl"
)

# Scale test data
X_test = scaler.transform(test_samples)

# Predict probabilities
probs = model.predict_proba(X_test)
preds = model.predict(X_test)

# Display results
for i, (p, pred) in enumerate(zip(probs, preds), 1):
    print(f"\nSample {i}:")
    for label, prob in zip(le.classes_, p):
        print(f"  {label.split('_')[0]:<8}: {prob:.3f}")
    print("➡️ Predicted:", le.inverse_transform([pred])[0].split('_')[0])
