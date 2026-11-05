"""Model loading and training setup extracted from the notebook run cells."""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from configs import EXPERIMENTS
from losses import DiceAndCrossEntropyLoss


def build_model(experiment, device=None):
    """Load the backbone and construct the manuscript-corrected segmentation model."""
    config = EXPERIMENTS[experiment]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    architecture = config["architecture"]
    kwargs = config["model"]

    if architecture == "Sen2DINOv3":
        from transformers import AutoModel
        from models.Sen2DINOv3 import Sen2DINOv3

        backbone = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-sat493m",
            torch_dtype=torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            backbone = backbone.to(device)
        model = Sen2DINOv3(backbone, **kwargs)
    elif architecture == "DOFA":
        from torchgeo.models import DOFALarge16_Weights, dofa_large_patch16_224
        from models.DOFA import DOFASegmentationUperNet

        backbone = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).to(device)
        model = DOFASegmentationUperNet(backbone, **kwargs)
    elif architecture == "Panopticon":
        from torchgeo.models import Panopticon_Weights, panopticon_vitb14
        from models.Panopticon import PanopticonSegmentationUperNet

        backbone = panopticon_vitb14(
            weights=Panopticon_Weights.VIT_BASE14, img_size=config["input_size"]
        ).to(device)
        model = PanopticonSegmentationUperNet(backbone, **kwargs)
    elif architecture == "MAE-SSL4EO":
        from models.MAE_SSL4EO import MAE_SSL4EO

        model = MAE_SSL4EO(**kwargs)
    elif architecture == "MAE-SSL4EO-LORA":
        from models.MAE_SSL4EO_LORA import MAE_SSL4EO_LORA

        model = MAE_SSL4EO_LORA(**kwargs)
    elif architecture == "DeepLabV3+":
        from models.DeepLabV3Plus import create_deeplabv3plus

        model = create_deeplabv3plus()
    elif architecture == "SegFormer":
        from models.SegFormer import create_segformer

        model = create_segformer()
    elif architecture == "U-Net++":
        from models.UNetPlusPlus import create_unetplusplus

        model = create_unetplusplus()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model.to(device)


def configure_training(model, config):
    """Create the manuscript loss/optimizer and the binary foreground metrics."""
    import torchmetrics
    from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryStatScores

    device = next(model.parameters()).device
    learning_rate = config["learning_rate"]
    if config["model"].get("use_lora", False):
        if config["architecture"] == "MAE-SSL4EO-LORA":
            decoder_params, lora_params = model.get_parameter_groups()
        else:
            decoder_params, lora_params = [], []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "lora_" in name or "lora_A" in name or "lora_B" in name:
                        lora_params.append(param)
                    else:
                        decoder_params.append(param)
        parameters = [
            {"params": decoder_params, "lr": learning_rate},
            {"params": lora_params, "lr": learning_rate},
        ]
    elif config["architecture"] == "MAE-SSL4EO":
        parameters = [param for param in model.parameters() if param.requires_grad]
    else:
        parameters = model.parameters()

    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=config["weight_decay"])
    scheduler_kwargs = {key: value for key, value in config["scheduler"].items() if key != "name"}
    scheduler_class = {"plateau": ReduceLROnPlateau, "cosine": CosineAnnealingLR}[config["scheduler"]["name"]]
    scheduler = scheduler_class(optimizer, **scheduler_kwargs)
    return {
        "loss_fn": DiceAndCrossEntropyLoss().to(device),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "metric": torchmetrics.Dice().to(device),
        "metric2": BinaryJaccardIndex().to(device),
        "metric3": BinaryF1Score().to(device),
        "stat_scores_metric": BinaryStatScores(threshold=0.5).to(device),
    }