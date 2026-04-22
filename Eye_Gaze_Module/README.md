# 👁️ Eye Gaze Emotion Recognition Module

This module predicts human emotional states using eye movement patterns and pupil behavior.

---

## 🔍 Overview

The system uses gaze coordinates and pupil dilation data to classify emotions into:

- Anger
- Disgust
- Fear
- Happy
- Sad

---

## ⚙️ Features Extracted

- avg_x → Average horizontal gaze position  
- avg_y → Average vertical gaze position  
- std_x → Horizontal variability  
- std_y → Vertical variability  
- avg_pupil → Average pupil diameter  
- saccade_speed → Eye movement speed  

---
## 📊 Dataset Collection

The eye-gaze dataset was collected at the **National Institute of Mental Health and Neurosciences (NIMHANS), Bengaluru** using a **Tobii Eye Tracker**.

### Data Details:
- Participants: 4 subjects
- Data format: Excel (.xlsx)
- Signals captured:
  - Left & right gaze coordinates
  - Pupil diameter
  - Timestamps
  - Stimulus ID

---

## ⚙️ Preprocessing Pipeline

The raw eye-tracking data was preprocessed using the following steps:

1. **Data Integration**
   - Combined multiple participant files into a single dataset

2. **Timestamp Processing**
   - Converted device and system timestamps to datetime format

3. **Data Cleaning**
   - Removed rows with missing gaze or pupil data

4. **Feature Engineering**
   - Average pupil diameter
   - Average gaze position (X, Y)
   - Extracted gaze coordinates from raw tuple strings

5. **Final Output**
   - Saved as: Data/preprocessed_gaze_data.csv

## 🧠 Models Used

| Model            | Accuracy |
|------------------|---------|
| Random Forest    | 90%     |
| CatBoost         | 94.92%  |
| XGBoost          | 95.00%  |
| MLP              | 97.62%  |

---

## 📊 Results

### Random Forest
<img src="Outputs/rf_confusion.png" width="400"/>

### XGBoost
<img src="Outputs/xgb_confusion.png" width="400"/>

### MLP
<img src="Outputs/mlp_confusion.png" width="400"/>

### CatBoost
<img src="Outputs/catboost_confusion.png" width="400"/>

---

## 🎯 Real-Time System (OCULUS)

- Integrated with **Tobii Eye Tracker**
- Live gaze tracking + prediction
- Heatmap + scanpath visualization
- Flask backend + HTML dashboard

---

## 🎥 Demo

<p align="center">
  <video src="Demo/realtime_demo.mp4" width="600" controls></video>
</p>

---

## 🚀 How to Run

```bash
pip install opencv-python flask joblib tobii_research python-vlc
python app_server_tobii.py
