import torch
import torch.nn as nn

from dtu_mlops.model import BasicBlock, Bottleneck, resnet18, resnet50, Model


# -----------------------------
# ResNet factory tests
# -----------------------------
def test_resnet18_builds_with_correct_classifier():
    """
    Checks that:
    - resnet18 builds successfully
    - the final classifier has the correct output size (num_classes)
    - conv1 expects the correct input channels (in_channels)
    """
    model = resnet18(num_classes=11, in_channels=1)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 11
    assert model.conv1.in_channels == 1


def test_resnet18_forward_output_shape():
    """
    Forward pass should produce (batch_size, num_classes)
    """
    model = resnet18(num_classes=11, in_channels=1)

    x = torch.randn(4, 1, 28, 28)
    y = model(x)

    assert y.shape == (4, 11)


def test_resnet50_forward_output_shape():
    """
    Same check for resnet50 (more layers, but output shape must match)
    """
    model = resnet50(num_classes=11, in_channels=1)

    x = torch.randn(2, 1, 28, 28)
    y = model(x)

    assert y.shape == (2, 11)


# -----------------------------
# Block behavior tests
# -----------------------------
def test_basicblock_no_downsample_preserves_shape():
    """
    If stride=1 and no downsample, BasicBlock should preserve shape.
    """
    block = BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None)

    x = torch.randn(1, 64, 28, 28)
    y = block(x)

    assert y.shape == x.shape


def test_basicblock_with_downsample_halves_spatial_dimensions():
    """
    If stride=2, output should shrink spatial dimensions (28->14),
    and downsample should be applied to identity to match shapes.
    """
    downsample = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(64),
    )
    block = BasicBlock(in_channels=64, out_channels=64, stride=2, downsample=downsample)

    x = torch.randn(1, 64, 28, 28)
    y = block(x)

    assert y.shape == (1, 64, 14, 14)


def test_bottleneck_expands_channels():
    """
    Bottleneck block expansion=4.
    So out_channels=64 should become 256 output channels.
    """
    downsample = nn.Sequential(
        nn.Conv2d(64, 64 * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64 * Bottleneck.expansion),
    )
    block = Bottleneck(in_channels=64, out_channels=64, stride=1, downsample=downsample)

    x = torch.randn(2, 64, 28, 28)
    y = block(x)

    assert y.shape == (2, 256, 28, 28)


# -----------------------------
# Weight init sanity test
# -----------------------------
def test_resnet_initializes_batchnorm_weight_and_bias():
    """
    _initialize_weights() sets BatchNorm weight=1 and bias=0.
    We'll check bn1 as a representative example.
    """
    model = resnet18(num_classes=11, in_channels=1)

    assert torch.allclose(model.bn1.weight.detach(), torch.ones_like(model.bn1.weight))
    assert torch.allclose(model.bn1.bias.detach(), torch.zeros_like(model.bn1.bias))


# -----------------------------
# Backwards compatibility Model
# -----------------------------
def test_dummy_model_forward_shape():
    """
    Your old 'Model' class is very simple:
      Linear(1 -> 1)
    Test that it runs.
    """
    model = Model()
    x = torch.randn(5, 1)
    y = model(x)
    assert y.shape == (5, 1)
