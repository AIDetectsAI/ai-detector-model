import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    """
    Two 3×3 convs with batchnorm and ReLU, plus skip connection.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x)
        return self.relu(out + skip)

class CustomBinaryCNN(nn.Module):
    """
    Custom CNN for AI vs. natural image classification.
    - 4 residual convolutional stages
    - SpatialDropout2d for regularization
    - Global average pooling
    - Small classification head
    """
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.stage1(x)    # [B,32,H/2,W/2]
        x = self.stage2(x)    # [B,64,H/4,W/4]
        x = self.stage3(x)    # [B,128,H/8,W/8]
        x = self.stage4(x)    # [B,256,H/16,W/16]
        x = self.global_pool(x)  # [B,256,1,1]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self(x)
            return self.sigmoid(x)