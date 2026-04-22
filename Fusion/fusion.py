# ==========================================================
# NeuroCanvas Fusion Dashboard Backend (Flask)
# ==========================================================

from flask import Flask, jsonify
import numpy as np
import random
import time

app = Flask(__name__)

# ----------------------------------------------------------
# EEG ZONE MAPPING (3 classes)
# ----------------------------------------------------------
EEG_PRIOR = {
    'NEG-LOW':  {'sad':0.75, 'anger':0.08, 'fear':0.10, 'disgust':0.04, 'happy':0.03},
    'NEG-HIGH': {'sad':0.07, 'anger':0.40, 'fear':0.43, 'disgust':0.06, 'happy':0.04},
    'MIXED':    {'sad':0.05, 'anger':0.25, 'fear':0.07, 'disgust':0.35, 'happy':0.28},
}

EMOTIONS = ['sad','anger','fear','disgust','happy']

# ----------------------------------------------------------
# SYNTHETIC DATA GENERATOR
# ----------------------------------------------------------
def generate_synthetic_data():
    return {
        "eeg_zone": random.choice(['NEG-LOW','NEG-HIGH','MIXED']),
        "eeg_conf": random.randint(70, 95),
        "eye_emotion": random.choice(EMOTIONS),
        "eye_conf": random.randint(65, 95),
        "delta": round(random.uniform(0.3, 1.0), 3),
        "theta": round(random.uniform(0.01, 0.2), 3),
        "alpha": round(random.uniform(0.001, 0.2), 4),
        "beta": round(random.uniform(0.0001, 0.1), 4),
        "pupil": round(random.uniform(2.8, 4.2), 2),
        "saccade": round(random.uniform(2.0, 6.0), 2),
        "x": round(random.uniform(0.2, 0.8), 3),
        "y": round(random.uniform(0.2, 0.8), 3)
    }

# ----------------------------------------------------------
# FUSION LOGIC (CORE)
# ----------------------------------------------------------
def fuse_emotions(eeg_zone, eeg_conf, eye_emotion, eye_conf):

    # Step 1: EEG prior
    prior = EEG_PRIOR[eeg_zone]

    # Step 2: Scale by EEG confidence
    eeg_weight = eeg_conf / 100
    scaled_prior = {
        e: eeg_weight * prior[e] + (1 - eeg_weight) * 0.2
        for e in EMOTIONS
    }

    # Step 3: Eye likelihood
    eye_weight = eye_conf / 100
    likelihood = {}

    for e in EMOTIONS:
        if e == eye_emotion:
            likelihood[e] = eye_weight * 0.80 + 0.10
        else:
            likelihood[e] = (1 - eye_weight * 0.80) / (len(EMOTIONS)-1)

    # Step 4: Joint probability
    joint = {}
    total = 0

    for e in EMOTIONS:
        joint[e] = scaled_prior[e] * likelihood[e]
        total += joint[e]

    # Normalize
    for e in EMOTIONS:
        joint[e] /= total

    # Step 5: Final prediction
    fused_emotion = max(joint, key=joint.get)
    confidence = round(joint[fused_emotion] * 100, 2)

    return fused_emotion, confidence, joint

# ----------------------------------------------------------
# API ENDPOINT
# ----------------------------------------------------------
@app.route("/get_emotion", methods=["GET"])
def get_emotion():

    data = generate_synthetic_data()

    fused_emotion, confidence, scores = fuse_emotions(
        data["eeg_zone"],
        data["eeg_conf"],
        data["eye_emotion"],
        data["eye_conf"]
    )

    response = {
        "timestamp": time.time(),
        "eeg": {
            "zone": data["eeg_zone"],
            "confidence": data["eeg_conf"],
            "bands": {
                "delta": data["delta"],
                "theta": data["theta"],
                "alpha": data["alpha"],
                "beta": data["beta"]
            }
        },
        "eye": {
            "emotion": data["eye_emotion"],
            "confidence": data["eye_conf"],
            "pupil": data["pupil"],
            "saccade": data["saccade"],
            "gaze": {
                "x": data["x"],
                "y": data["y"]
            }
        },
        "fusion": {
            "emotion": fused_emotion,
            "confidence": confidence,
            "distribution": scores
        }
    }

    return jsonify(response)

# ----------------------------------------------------------
# RUN SERVER
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)