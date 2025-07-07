from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile
import librosa
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import sys

# Fix the path separator issue
sys.path.append(os.path.join('backend', 'model.py'))
from model import ECAPA_gender
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Voice Analysis API",
              description="API for voice health and gender classification")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS, images)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main HTML page
@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    if os.path.exists("index.html"):
        return FileResponse('index.html')
    elif os.path.exists("frontend/index.html"):
        return FileResponse('frontend/index.html')
    else:
        return {
            "message": "Voice Analysis API is running!",
            "endpoints": {
                "health_check": "/health",
                "predict": "/predict (POST)",
                "documentation": "/docs"
            },
            "note": "Place your index.html file in the root directory or create a 'frontend' folder"
        }

class RepeatChannels(tf.keras.layers.Layer):
    def __init__(self, repeats, axis, **kwargs):
        super(RepeatChannels, self).__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs):
        return tf.repeat(inputs, repeats=self.repeats, axis=self.axis)

    def get_config(self):
        config = super(RepeatChannels, self).get_config()
        config.update({"repeats": self.repeats, "axis": self.axis})
        return config

# Load voice health model (modify paths as needed)
VOICE_HEALTH_MODEL_PATH = "models/optimized_fusion_model.keras"
VOICE_SCALER_PATH = "models/scaler.pkl"
VOICE_LABEL_ENCODER_PATH = "models/label_encoder.pkl"
VOICE_THRESHOLD_PATH = "models/optimal_threshold.pkl"

# Initialize models as None
voice_model = None
voice_scaler = None
voice_label_encoder = None
voice_threshold = None
gender_model = None

try:
    # Register custom layer and load model
    custom_objects = {"RepeatChannels": RepeatChannels}
    voice_model = load_model(VOICE_HEALTH_MODEL_PATH, custom_objects=custom_objects)
    voice_scaler = joblib.load(VOICE_SCALER_PATH)
    voice_label_encoder = joblib.load(VOICE_LABEL_ENCODER_PATH)
    voice_threshold = joblib.load(VOICE_THRESHOLD_PATH)
    print("Voice health model loaded successfully!")
except Exception as e:
    print(f"Error loading voice health model: {str(e)}")

# --- Gender Classification Model Setup ---
try:
    gender_model = ECAPA_gender.from_pretrained('JaesungHuh/ecapa-gender')
    gender_model.eval()
    print("Gender model loaded successfully!")
except Exception as e:
    print(f"Error loading gender model: {str(e)}")

# --- Feature Extraction Functions ---
def extract_enhanced_voice_features(y, sr, min_length=2048):
    if len(y) < min_length:
        return np.zeros(60)  # Expanded feature set
    
    features = []
    
    # Amplitude and energy features
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))
    features.append(np.max(rms))
    
    # Temporal features
    zero_crossings = librosa.feature.zero_crossing_rate(y=y)[0]
    features.append(np.mean(zero_crossings))
    features.append(np.std(zero_crossings))
    
    # Spectral shape features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))
    
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(spec_bandwidth))
    features.append(np.std(spec_bandwidth))
    
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(spec_rolloff))
    features.append(np.std(spec_rolloff))
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.append(np.mean(np.mean(contrast, axis=1)))
    features.append(np.std(np.mean(contrast, axis=1)))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.append(np.mean(mfccs[i]))
        features.append(np.std(mfccs[i]))
    
    # Voice stability features
    if len(y) >= sr * 0.5:
        y_segments = []
        segment_length = int(sr * 0.1)
        for i in range(0, len(y) - segment_length, segment_length):
            y_segments.append(y[i:i+segment_length])
        
        if len(y_segments) >= 3:
            segment_mfccs = [np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=5), axis=1) for seg in y_segments]
            mfcc_diffs = []
            for i in range(1, len(segment_mfccs)):
                mfcc_diffs.append(np.mean(np.abs(segment_mfccs[i] - segment_mfccs[i-1])))
            features.append(np.mean(mfcc_diffs))
    else:
        features.append(0)
        
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    if np.any(magnitudes > 0):
        pitch_max_indices = np.argmax(magnitudes, axis=0)
        pitches_max = np.array([pitches[pitch_max_indices[i], i] for i in range(pitches.shape[1])])
        pitches_max = pitches_max[pitches_max > 0]
        if len(pitches_max) > 0:
            features.append(np.mean(pitches_max))
            features.append(np.std(pitches_max))
            features.append(np.median(pitches_max))
            if len(pitches_max) > 1:
                features.append(np.mean(np.abs(np.diff(pitches_max))))
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])
    
    # Harmonics to noise ratio
    S = np.abs(librosa.stft(y))
    harmonics = np.mean(S, axis=1)
    noise = np.std(S, axis=1)
    if np.sum(noise) > 0:
        hnr = np.sum(harmonics) / np.sum(noise)
        features.append(hnr)
    else:
        features.append(0)
    
    return np.array(features)

# --- Prediction Functions ---
def predict_voice_health(audio_path: str) -> Dict[str, Any]:
    """Predict voice health status"""
    if voice_model is None:
        raise HTTPException(status_code=503, detail="Voice health model not loaded")
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        if len(y) < 2048:
            return {"error": "Audio too short"}
        
        # Create spectrogram for CNN branch
        D = librosa.stft(y)
        spectrogram = librosa.amplitude_to_db(np.abs(D))
        spectrogram_resized = cv2.resize(spectrogram, (224, 224))
        spectrogram_resized = np.expand_dims(spectrogram_resized, axis=-1)
        X_spec = np.expand_dims(spectrogram_resized, axis=0)
        
        # Extract engineered features for feature branch
        features = extract_enhanced_voice_features(y, sr)
        features_scaled = voice_scaler.transform(features.reshape(1, -1))
        
        # Generate prediction
        prediction = voice_model.predict([X_spec, features_scaled])
        
        # Apply optimal threshold
        predicted_class = 1 if prediction[0, 1] >= voice_threshold else 0
        class_name = voice_label_encoder.inverse_transform([predicted_class])[0]
        
        # Calculate confidence
        confidence = prediction[0, predicted_class]
        
        return {
            "status": class_name,
            "confidence": float(confidence),
            "healthy_prob": float(prediction[0, 0]),
            "unhealthy_prob": float(prediction[0, 1])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice health prediction error: {str(e)}")

def predict_gender(audio_path: str) -> Dict[str, float]:
    """Predict gender from voice"""
    if gender_model is None:
        raise HTTPException(status_code=503, detail="Gender model not loaded")
    
    try:
        audio = gender_model.load_audio(audio_path)
        with torch.no_grad():
            output = gender_model.forward(audio)
            probs = torch.softmax(output, dim=1)
            prob_dict = {gender_model.pred2gender[i]: float(prob) for i, prob in enumerate(probs[0])}
        return prob_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gender prediction error: {str(e)}")

# --- API Endpoints ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for combined voice analysis"""
    # Validate file type
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving temporary file: {str(e)}")
    
    try:
        # Run both predictions
        health_result = predict_voice_health(temp_path)
        gender_result = predict_gender(temp_path)
        
        # Determine final gender (male/female with highest probability)
        final_gender = max(gender_result.items(), key=lambda x: x[1])[0]
        
        # Combine results
        result = {
            "status": health_result["status"],
            "gender": final_gender,
            "gender_probabilities": gender_result,
            "health_confidence": health_result["confidence"],
            "health_probabilities": {
                "healthy": health_result["healthy_prob"],
                "unhealthy": health_result["unhealthy_prob"]
            }
        }
        
        return result
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = {
        "voice_health_model": voice_model is not None,
        "gender_model": gender_model is not None
    }
    return {
        "status": "OK",
        "models_loaded": models_loaded
    }

# API endpoint for frontend to get model status
@app.get("/api/status")
async def api_status():
    """API status endpoint for frontend"""
    return {
        "api_status": "running",
        "models": {
            "voice_health": voice_model is not None,
            "gender_classification": gender_model is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)