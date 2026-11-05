"""SegFormer with the MiT-B3 encoder confirmed by the author."""

import segmentation_models_pytorch as smp


def create_segformer():
    return smp.Segformer(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        in_channels=12,
        classes=1,
    )