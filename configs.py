"""Manuscript settings, supplemented by notebook details absent from the paper."""

STANDARD_LR = 0.001
LORA_LR = 0.0003

_UPERNET = {"in_channels": 12, "num_classes": 1, "fpn_channels": 256}
_LORA = {"use_lora": True, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.05}
_PLATEAU = {"name": "plateau", "mode": "min", "factor": 0.5, "patience": 5}
_ACCUMULATED = {
    "variant": "accumulated",
    "num_epochs": 60,
    "patience": 10,
    "accumulation_steps": 1,
    "batch_size": 32,
    "learning_rate": LORA_LR,
    "weight_decay": 1e-4,
    "scheduler": _PLATEAU,
}

EXPERIMENTS = {
    "Sen2DINOv3": {
        **_ACCUMULATED,
        "architecture": "Sen2DINOv3",
        "input_size": 512,
        "model": {**_UPERNET, **_LORA},
        "checkpoint_name": "best_model_dinov3_upernet_512x512.pt",
    },
    "DOFA-LoRA": {
        **_ACCUMULATED,
        "architecture": "DOFA",
        "input_size": 512,
        "model": {"num_classes": 1, "fpn_channels": 256, **_LORA},
        "checkpoint_name": "best_model_dofa_upernet_512x512.pt",
    },
    "Panopticon-LoRA": {
        **_ACCUMULATED,
        "architecture": "Panopticon",
        "input_size": 518,
        "model": {**_UPERNET, **_LORA},
        "checkpoint_name": "best_model_panoptican_518x518.pt",
    },
    "MAE-SSL4EO": {
        "architecture": "MAE-SSL4EO",
        "input_size": 224,
        "model": {**_UPERNET, "freeze_encoder": True},
        "variant": "mae",
        "num_epochs": 10,
        "patience": 5,
        "accumulation_steps": 1,
        "batch_size": 32,
        "learning_rate": STANDARD_LR,
        "weight_decay": 1e-4,
        "scheduler": _PLATEAU,
        "checkpoint_name": "best_mae_upernet.pt",
    },
    "MAE-SSL4EO-LORA": {
        "architecture": "MAE-SSL4EO-LORA",
        "input_size": 224,
        "model": {**_UPERNET, **_LORA},
        "variant": "mae",
        "num_epochs": 50,
        "patience": 10,
        "accumulation_steps": 1,
        "batch_size": 32,
        "learning_rate": LORA_LR,
        "weight_decay": 1e-4,
        "scheduler": _PLATEAU,
        "checkpoint_name": "best_mae_upernet_lora.pt",
    },
    "DeepLabV3+": {
        "architecture": "DeepLabV3+",
        "input_size": 224,
        "model": {},
        "variant": "baseline",
        "num_epochs": 50,
        "patience": 5,
        "accumulation_steps": 1,
        "batch_size": 32,
        "learning_rate": STANDARD_LR,
        "weight_decay": 0.0,
        "scheduler": _PLATEAU,
        "checkpoint_name": "best_model_version_DeepLabv3+_resnet50.pt",
    },
    "SegFormer": {
        "architecture": "SegFormer",
        "input_size": 224,
        "model": {},
        "variant": "baseline",
        "num_epochs": 50,
        "patience": 10,
        "accumulation_steps": 1,
        "batch_size": 32,
        "learning_rate": STANDARD_LR,
        "weight_decay": 0.0,
        "scheduler": {"name": "cosine", "T_max": 50, "eta_min": 1e-6},
        "checkpoint_name": "best_model_version_Segformer_v02.pt",
    },
    "U-Net++": {
        "architecture": "U-Net++",
        "input_size": 224,
        "model": {},
        "variant": "unet",
        "num_epochs": 80,
        "patience": 15,
        "accumulation_steps": 1,
        "batch_size": 32,
        "learning_rate": STANDARD_LR,
        "weight_decay": 0.01,
        "scheduler": {
            "name": "plateau", "mode": "min", "factor": 0.8,
            "patience": 7, "min_lr": 1e-6,
        },
        "checkpoint_name": "best_model_unet++_efficientnet-b8.pt",
    },
}

# Same architecture and settings; the article's DINOv3-LoRA run uses 224 inputs.
EXPERIMENTS["DINOv3-LoRA"] = {**EXPERIMENTS["Sen2DINOv3"], "input_size": 224}