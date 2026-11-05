"""Original SMP constructor from unet++_resnet50.ipynb, cell 11."""

import segmentation_models_pytorch as smp


def create_unetplusplus():
    return smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=12,
        classes=1,
    )