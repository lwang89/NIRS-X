from torch import nn
from torch import unsqueeze, squeeze
import torch
import math
from einops import repeat


class CNNEncoder(nn.Module):
    "Use spatial and temporal convolutions to extract spatial-temporal embeddings."

    def __init__(self, num_classes, dropout=0.5):
        super().__init__()

        # Spatial CNN
        self.spatial = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=(8, 1)),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )

        # Temporal CNN
        self.temporal = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=(1, 3)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(128, kernel_size=(1, 5)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Linear classifier
        self.linear_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        x = unsqueeze(x, 1).permute(0, 1, 3, 2)
        x = self.spatial(x)
        x = self.temporal(x)

        # Comment this out for contrastive learning (NIRSiam) or NIRSformer
        # x = self.linear_classifier(x)

        return x

class PositionalEncoding(nn.Module):
    """ Copied from pytorch"""
    def __init__(self, d_model=128, dropout=0.1, max_len=145):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "")

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.to(self.device)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NIRSformer(nn.Module):
    def __init__(self, num_classes, dropout=0.1, d_model=128, nhead=8, nlayers=6):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "")

        # Spatial temporal encoder
        self.cnn_encoder = CNNEncoder(num_classes, dropout)

        # Classification token, copied from fNIRS-Transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)).to(self.device)

        # Positional encoding
        self.pos_encoder = PositionalEncoding()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=64)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Linear classifier
        self.linear_classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

        # Sequence feature number
        self.d_model = d_model

    def forward(self, x):
        x = self.cnn_encoder(x) * math.sqrt(self.d_model)
        x = squeeze(x).permute(2,0,1)
        _, batch_size, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'n () d -> n b d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=0)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        x = x[:, 0, :]

        # Comment this out for contrastive learning (NIRSiam)
        x = self.linear_classifier(x)

        return x
