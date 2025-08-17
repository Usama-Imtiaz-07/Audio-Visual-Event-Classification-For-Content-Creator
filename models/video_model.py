import torchvision.models as models
import torch.nn as nn
import torch


class VideoFeatureExtractor:
    def __init__(self):
        # Loading ResNet18 pre-trained on ImageNet
        self.vision_model = models.resnet18(pretrained=True)
        # Remove final classifier, keep feature extractor
        self.vision_model = nn.Sequential(*list(self.vision_model.children())[:-1])
    
    def extract_features(self, image_tensor):
        # image -> embeddings
        with torch.no_grad():
            features = self.vision_model(image_tensor.unsqueeze(0))
        return features.view(features.size(0), -1)  # Flatten