# =======================
# 1. Import Libraries
# =======================
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =======================
# 2. Load Saved Model & Encoder
# =======================
print("Loading model...")

model = joblib.load("eye_gaze_model.pkl")      # trained model
le = joblib.load("label_encoder.pkl")          # label encoder

print("✅ Model and encoder loaded successfully!")

# =======================
# 3. Load Test Dataset
# =======================
# Replace with your test file path
test_df = pd.read_csv("test_gaze_data.csv")

print("\nTest Data Shape:", test_df.shape)
print(test_df.head())

# =======================
# 4. Feature Selection
# =======================
features = ['avg_x', 'avg_y', 'std_x', 'std_y', 'avg_pupil', 'saccade_speed']

# Check if all required features exist
missing = [f for f in features if f not in test_df.columns]
if missing:
    raise ValueError(f"Missing columns in test data: {missing}")

X_test = test_df[features]

# =======================
# 5. Prediction
# =======================
print("\nPredicting...")

y_pred = model.predict(X_test)

# Convert numeric labels → original emotion names
predicted_emotions = le.inverse_transform(y_pred)

# Add predictions to dataframe
test_df['Predicted_Emotion'] = predicted_emotions

print("\nSample Predictions:")
print(test_df[['Predicted_Emotion']].head())

# =======================
# 6. Save Predictions
# =======================
output_file = "predicted_results.csv"
test_df.to_csv(output_file, index=False)

print(f"\n✅ Predictions saved to {output_file}")

# =======================
# 7. Evaluation (Optional)
# =======================
if 'Emotion' in test_df.columns:
    print("\nEvaluating model performance...")

    # Convert actual labels to numeric
    y_true = le.transform(test_df['Emotion'])

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print("\n🎯 Test Accuracy:", acc)

    # Classification Report
    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

else:
    print("\n⚠️ No true labels found. Skipping evaluation.")

# =======================
# 8. Single Sample Prediction (Optional)
# =======================
print("\nTesting single sample prediction...")

sample = np.array([[0.5, 0.4, 0.1, 0.1, 3.2, 0.8]])  # Replace with real values
sample_pred = model.predict(sample)

print("Predicted Emotion for sample:",
      le.inverse_transform(sample_pred)[0])