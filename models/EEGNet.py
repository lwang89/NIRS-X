import torch
from torch import nn
from torch.nn import functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)

        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self, feature_size=8, window_size=150, num_classes=2, F1=4, D=2, F2=8, avgpool2d_1=4, avgpool2d_2=8, dropout=0.5):
        super(EEGNet, self).__init__()
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

        self.classifier = nn.Sequential(
            nn.Linear(in_features=F2 * (window_size//(avgpool2d_1*avgpool2d_2)), out_features=num_classes, bias=True)
        )  

    def forward(self, x):
        x = self.firstConv(x.unsqueeze(1).transpose(2, 3))
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        
        # Comment this out for contrastive learning (NIRSiam)
        x = self.classifier(x)
        normalized_probabilities = F.log_softmax(x, dim=1)

        # 1. For EEGNet and DeepConvNet, directly use nn.NLLLoss() as criterion; 
        # 2. Return x for contrastive learning (NIRSiam)
        return normalized_probabilities  
