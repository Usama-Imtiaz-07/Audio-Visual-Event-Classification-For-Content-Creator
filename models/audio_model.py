

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

class AudioFeatureExtractor:
    def __init__(self):
        # load wav2vec processor & model
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    def extract_features(self, audio_waveform, sampling_rate=16000):
        # audio -> embeddings 
        inputs = self.processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.audio_model(**inputs).last_hidden_state.mean(dim=1)
        return features

     
