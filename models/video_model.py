import torchvision.models as models
import torch.nn as nn
import torch


class VideoFeatureExtractor:
    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Loading ResNet18 pre-trained on ImageNet
        self.vision_model = models.resnet18(pretrained=False)

        # Load weights manually
        state_dict = torch.load("./checkpoints/resnet18-f37072fd.pth", map_location=torch.device("cpu"))
        self.vision_model.load_state_dict(state_dict)

        # Remove final classifier, keep feature extractor
        self.vision_model = nn.Sequential(*list(self.vision_model.children())[:-1])
        self.vision_model.to(self.device)

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: [3, H, W] already normalized to ImageNet stats
        returns: [1, 512] feature vector
        """
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        with torch.no_grad():
            features = self.vision_model(image_tensor)  # [1, 512, 1, 1]
        return features.view(features.size(0), -1)  # flatten to [1, 512]