"""DeepLabV3+ ResNet-50 from the notebook; the ignored SCSE keyword is omitted."""

import segmentation_models_pytorch as smp


def create_deeplabv3plus():
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=12,
        classes=1,
    )