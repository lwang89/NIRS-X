import torch
from torch import nn
from torch.nn import functional as F
from models.base_models import base_DeepConvNet


class DeepConvNet(nn.Module):
    def __init__(self, feature_size=8, num_timesteps=150, num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = base_DeepConvNet(feature_size, num_timesteps, num_classes, dropout)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=num_classes, kernel_size=(1, 5), stride=1, bias=True)
        )

    def forward(self, x):
        x = self.embedding(x)

        # Comment this out for contrastive learning (NIRSiam)
        x = self.classifier(x) 
        x = x.squeeze(dim=2).squeeze(dim=2)
        normalized_probabilities = F.log_softmax(x, dim=1)
        
        # 1. For EEGNet and DeepConvNet, directly use nn.NLLLoss() as criterion; 
        # 2. Return x for contrastive learning (NIRSiam)
        return normalized_probabilities  
