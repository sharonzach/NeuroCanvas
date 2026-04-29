import joblib
import pandas as pd
from catboost import Pool

# Load trained model
model = joblib.load("catboost_model.pkl")

# Same sample data
sample = pd.DataFrame([{
    'avg_x': 0.5,
    'avg_y': 0.3,
    'std_x': 0.1,
    'std_y': 0.1,
    'avg_pupil': 3.2,
    'saccade_speed': 4.0,
    'Subject': 'S1',
    'stimulus_id': 'img1'
}])

# Define categorical columns (SAME as training)
cat_features = ['Subject', 'stimulus_id']

# Convert to CatBoost Pool
sample_pool = Pool(sample, cat_features=cat_features)

# Predict
prediction = model.predict(sample_pool)

print("Predicted Emotion:", prediction[0])