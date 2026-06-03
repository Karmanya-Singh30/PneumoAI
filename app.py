from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)
# This is the magic line that allows your HTML file to talk to this Python file
CORS(app) 

# Load the saved model and scaler
rf_model = joblib.load('pneumo_rf_model.pkl')
scaler = joblib.load('pneumo_scaler.pkl')
BEST_THRESHOLD = 0.30

# Your exact feature extraction function
def extract_features_per_cycle(file_path, sr=16000, cycle_duration=3.0):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    cycle_len = int(cycle_duration * sample_rate)
    cycles = [audio[i:i+cycle_len] for i in range(0, len(audio)-cycle_len, cycle_len)]
    if len(cycles) == 0: cycles = [audio]
    
    all_features = []
    for cycle in cycles:
        if len(cycle) < sample_rate * 0.1: continue
        mfcc = librosa.feature.mfcc(y=cycle, sr=sample_rate, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc, width=5), axis=1) if mfcc.shape[1] >= 5 else np.zeros(20)
        mfcc_delta2_mean = np.mean(librosa.feature.delta(mfcc, order=2, width=5), axis=1) if mfcc.shape[1] >= 5 else np.zeros(20)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=cycle, sr=sample_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=cycle, sr=sample_rate))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=cycle))
        rms = np.mean(librosa.feature.rms(y=cycle))
        
        feature_vector = np.hstack([
            mfcc_mean, mfcc_delta_mean, mfcc_delta2_mean,
            [spectral_centroid, spectral_bandwidth, zcr, rms],
            [0, 0] # crackles, wheezes
        ])
        all_features.append(feature_vector)
        
    return np.mean(all_features, axis=0).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path) # Save temporarily for librosa to read
    
    try:
        # Extract, Scale, Predict
        features = extract_features_per_cycle(file_path)
        features_scaled = scaler.transform(features)
        
        proba = rf_model.predict_proba(features_scaled)[0]
        normal_idx = list(rf_model.classes_).index('Normal')
        abnormal_idx = list(rf_model.classes_).index('Abnormal')
        
        normal_prob = proba[normal_idx] * 100
        abnormal_prob = proba[abnormal_idx] * 100
        
        prediction = 'Normal' if (normal_prob / 100) >= BEST_THRESHOLD else 'Abnormal'
        
        # Clean up temp file
        os.remove(file_path)
        
        # Send data back to the frontend
        return jsonify({
            'prediction': prediction,
            'normal_confidence': round(normal_prob, 1),
            'abnormal_confidence': round(abnormal_prob, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(port=5000, debug=True)