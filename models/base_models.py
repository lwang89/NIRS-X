import torch
from torch import nn
from torch.nn import functional as F

"""
base encoders here
"""


class base_Classifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.head(x)


class linear_classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.linear = nn.LazyLinear(self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        proj, pred = self.encoder(x)
        return self.linear(proj)


class base_DeepConvNet(nn.Module):
    def __init__(self, feature_size=8, num_timesteps=150, num_classes=2, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5), stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(feature_size, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),  
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), 
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),  
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

    def forward(self, x):
        x = self.block1(x.unsqueeze(1).transpose(2, 3))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x


class base_EEGNet(nn.Module):
    def __init__(self, feature_size=8, window_size=150, num_classes=2, F1=4, D=2, F2=8, avgpool2d_1=4, avgpool2d_2=8, dropout=0.5):
        super().__init__()
        # Temporal convolution
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(num_features=F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, kernel_size=(feature_size, 1), stride=(1, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), 
            nn.AvgPool2d(kernel_size=(1, avgpool2d_1)),  
            nn.Dropout(p=dropout)
        )

        # depthwise convolution follow by pointwise convolution (pointwise convolution is just Conv2d with 1x1 kernel)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),  
            nn.AvgPool2d(kernel_size=(1, avgpool2d_2)), 
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.firstConv(x.unsqueeze(1).transpose(2, 3))
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        return x




