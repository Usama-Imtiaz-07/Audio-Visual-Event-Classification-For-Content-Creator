from models.audio_model import AudioFeatureExtractor
from models.video_model import VideoFeatureExtractor
import librosa
import cv2
from PIL import Image
import torchvision.transforms as T


# initialize models
audio_model = AudioFeatureExtractor()
video_model = VideoFeatureExtractor()


# ---- Audio Feature Extraction ----

# load audio
audio_path = "./data/samples/audio/03-01-01-01-01-01-01.wav"
waveform, sr = librosa.load(audio_path, sr=16000)

audio_features = audio_model.extract_features(waveform, sr)
print("Audio feature shape:", audio_features.shape)

# ---- Video Feature Extraction ----

# Preprocess pipeline for ResNet
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# read one frame
cap = cv2.VideoCapture("./data/samples/video/01-01-01-01-01-02-01.mp4")
ret, frame = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)
img_tensor = preprocess(pil_img)

features = video_model.extract_features(img_tensor)
print("Features:", features.shape)  # should be [1, 512]