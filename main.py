from models.audio_model import AudioFeatureExtractor
from models.video_model import VideoFeatureExtractor
import torch
import librosa

# initialize models
audio_model = AudioFeatureExtractor()
#video_model = VideoFeatureExtractor()

# ---- Audio Feature Extraction ----

# load audio
audio_path = "./data/audio/03-01-01-01-01-01-01.wav"
waveform, sr = librosa.load(audio_path, sr=16000)

audio_features = audio_model.extract_features(waveform, sr)
print("Audio feature shape:", audio_features.shape)

# ---- Video Feature Extraction ----

