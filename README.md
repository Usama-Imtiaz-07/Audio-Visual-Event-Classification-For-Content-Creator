# Audio-Visual Event Classification for Content Creators 🎬🎙️

## Overview

This project explores **multimodal AI** for content creators, combining **audio** and **video** streams to automatically classify events and provide intelligent insights.  
The goal is to demonstrate how **audio-visual understanding** can be integrated into tools like **RODE Connect** or creator workflows, helping streamline editing, improve accessibility, and enable smarter content recommendations.

---

## Key Features

- **Audio Feature Extraction**  
  Using [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) to extract robust embeddings from raw speech/audio.

- **Video Feature Extraction**  
  Pretrained [ResNet18](https://pytorch.org/vision/stable/models/resnet.html) to capture spatial/visual features from video frames.

- **Feature Fusion (In Progress)**  
  Audio and video embeddings will be combined for event-level classification.

- **Optimized Inference (Planned)**  
  CUDA kernels, TensorRT optimizations, and quantization for efficient deployment on edge devices (e.g., audio gear, mobile apps).

---

## Example Use Case

Imagine a creator recording a podcast with video:  

- Detects laughter, applause, or emphasis automatically.  
- Identifies visual cues (hand gestures, expressions) aligned with audio events.  
- Provides real-time markers for editing software or metadata enrichment.  

This would allow product ecosystem (hardware + software) to deliver **next-gen creator tools** that are intelligent, adaptive, and workflow-friendly.

---

## Project Status

- ✅ Audio feature extraction working  
- ✅ Video feature extraction working  
- 🚧 Fusion + classifier under development  
- 🚀 Next step: low-level optimization with CUDA/TensorRT  

---

## Tech Stack

- **PyTorch** (deep learning framework)  
- **Torchvision / Hugging Face Transformers** (pretrained models)  
- **Librosa** (audio preprocessing)  
- **CUDA / TensorRT (planned)** for inference optimization  
- **Gradio** for User interface

---

## Repository Structure

├── models/
│ ├── audio_model.py # Audio feature extractor (Wav2Vec2)
│ ├── video_model.py # Video feature extractor (ResNet18)
├── notebooks/ # Experimentation/Prototyping
├── UI/ # Gradio app for frontend/UI
├── main.py # Core pipeline entrypoint / fusing models
├── test.py # Debug/testing scripts
├── requirements.txt
└── README.md

---

## Next Steps

- [ ] Implement multimodal fusion network  
- [ ] Train on small benchmark dataset (AudioSet / CREMA-D subset)  
- [ ] Apply CUDA/TensorRT optimizations  
- [ ] Benchmark latency for real-time use cases  

---

## Author

**Usama Imtiaz** – AI Engineer | Multimodal AI | Audio-Visual ML
