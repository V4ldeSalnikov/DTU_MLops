import torch
from torch import nn


class BasicBlock(nn.Module):
    """Basic building block for ResNet-18"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck building block for ResNet-50"""
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model for image classification"""

    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 11,
        in_channels: int = 1,
    ):
        super().__init__()
        self.in_channels = 64

        # Initial convolution layer (adapted for small 28x28 images)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Removed maxpool for small images

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block: type[BasicBlock | Bottleneck], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 11, in_channels: int = 1) -> ResNet:
    """ResNet-18 model
    
    Args:
        num_classes: Number of output classes (default: 11 for organamnist)
        in_channels: Number of input channels (default: 1 for grayscale)
    
    Returns:
        ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def resnet50(num_classes: int = 11, in_channels: int = 1) -> ResNet:
    """ResNet-50 model
    
    Args:
        num_classes: Number of output classes (default: 11 for organamnist)
        in_channels: Number of input channels (default: 1 for grayscale)
    
    Returns:
        ResNet-50 model
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


# Kept the old Model class for backward compatibility
class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


if __name__ == "__main__":
    # Test ResNet-18
    model18 = resnet18(num_classes=11, in_channels=1)
    x = torch.rand(4, 1, 28, 28)  # Batch of 4 grayscale 28x28 images
    output = model18(x)
    print(f"ResNet-18 output shape: {output.shape}")  # Should be [4, 11]
    
    # Test ResNet-50
    model50 = resnet50(num_classes=11, in_channels=1)
    output50 = model50(x)
    print(f"ResNet-50 output shape: {output50.shape}")  # Should be [4, 11]
    
    # Count parameters
    params18 = sum(p.numel() for p in model18.parameters())
    params50 = sum(p.numel() for p in model50.parameters())
    print(f"ResNet-18 parameters: {params18:,}")
    print(f"ResNet-50 parameters: {params50:,}")
