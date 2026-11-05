"""Sen2DINOv3, extracted from DINO_V3_UperNet_decoder_peft_512_512.ipynb."""

import copy

import numpy as np
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model

from .upernet import initialize_upernet_decoder, decode_upernet
from .adapters import DinoSpectralAdapter as SentinelSpectralAdapter


class Sen2DINOv3(nn.Module):

    def __init__(self, dino_backbone, in_channels=12, num_classes=1, use_lora=True,
                 lora_r=32, lora_alpha=32, lora_dropout=0.1, fpn_channels=256):
        super().__init__()

        self.adapter = SentinelSpectralAdapter(in_channels=in_channels, out_channels=3)

        self.encoder = copy.deepcopy(dino_backbone)

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print("LoRA applied to DINO-V3 encoder")
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'hidden_size'):
            embed_dim = self.encoder.config.hidden_size
        else:
            if hasattr(self.encoder, 'get_base_model'):
                base_model = self.encoder.get_base_model()
                if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                    embed_dim = base_model.config.hidden_size
                else:
                    embed_dim = 1024
            else:
                embed_dim = 1024

        self.skip_indices = [5, 11, 17, 23]

        initialize_upernet_decoder(self, embed_dim, fpn_channels, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor [B, 12, H, W] (12-channel Sentinel-2)
        Returns:
            Output tensor [B, num_classes, H, W]
        """

        x_in = self.adapter(x)

        outputs = self.encoder(
            pixel_values=x_in,
            output_hidden_states=True,
            interpolate_pos_encoding=True
        )

        if hasattr(outputs, 'hidden_states'):
            all_states = outputs.hidden_states
        else:
            all_states = outputs[0] if isinstance(outputs, tuple) else outputs

        raw_skips = [all_states[i + 1] for i in self.skip_indices]

        processed_skips = []
        for feat in raw_skips:

            patch_tokens = feat[:, 5:, :]
            B, N, C = patch_tokens.shape
            H = W = int(np.sqrt(N))

            processed_skips.append(patch_tokens.permute(0, 2, 1).reshape(B, C, H, W))

        return decode_upernet(self, processed_skips, x.shape[2:])

    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("Model Parameter Summary (Sen2DINOv3 with LoRA)")
        print("=" * 70)
        print(f"{'Component':<20} {'Total':>15} {'Trainable':>15} {'%':>10}")
        print("-" * 70)

        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        adapter_trainable = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"{'Adapter':<20} {adapter_params:>15,} {adapter_trainable:>15,} {adapter_trainable / adapter_params * 100:>9.2f}%")

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"{'Encoder (LoRA)':<20} {enc_params:>15,} {enc_trainable:>15,} {enc_trainable / enc_params * 100:>9.2f}%")

        dec_modules = [self.lateral_projections, self.ppm, self.fpn_top,
                       self.fpn_blocks, self.fpn_fusion, self.final_conv]
        dec_params = sum(p.numel() for m in dec_modules for p in m.parameters())
        dec_trainable = sum(p.numel() for m in dec_modules for p in m.parameters() if p.requires_grad)
        print(f"{'Decoder (UperNet)':<20} {dec_params:>15,} {dec_trainable:>15,} {dec_trainable / dec_params * 100:>9.2f}%")

        print("-" * 70)
        print(f"{'TOTAL':<20} {total_params:>15,} {trainable_params:>15,} {trainable_params / total_params * 100:>9.2f}%")
        print("=" * 70)