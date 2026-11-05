"""MAE-SSL4EO-LORA, extracted from MAE_ViT_Large_UperNet_Lora.ipynb."""

import numpy as np
from torch import nn
from peft import LoraConfig, get_peft_model
import timm
from torchgeo.models import ViTLarge16_Weights

from .upernet import initialize_upernet_decoder, decode_upernet
from .adapters import MAESpectralAdapter as SentinelSpectralAdapter


class MAE_SSL4EO_LORA(nn.Module):
    """MAE ViT-Large with the original 12-to-13 adapter, LoRA, and UPerNet."""

    def __init__(
        self,
        in_channels=12,
        num_classes=1,
        use_lora=True,
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        fpn_channels=256,
        mae_weights=ViTLarge16_Weights.SENTINEL2_ALL_MAE
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_lora = use_lora

        self.adapter = SentinelSpectralAdapter(
            in_channels=in_channels,
            out_channels=13
        )

        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            in_chans=13,
            num_classes=0,
            global_pool='',
        )

        self.encoder.load_state_dict(
            mae_weights.get_state_dict(progress=True),
            strict=False
        )
        print(f"MAE weights loaded from: {mae_weights}")

        if use_lora:

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["attn.qkv", "attn.proj"],
                bias="none",
                modules_to_save=None,
            )

            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"LoRA applied to MAE ViT-Large encoder")
            print(f"  - LoRA rank (r): {lora_r}")
            print(f"  - LoRA alpha: {lora_alpha}")
            print(f"  - LoRA dropout: {lora_dropout}")
            print(f"  - Target modules: attn.qkv, attn.proj")
        else:

            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (feature extraction mode)")

        if hasattr(self.encoder, 'get_base_model'):
            base_model = self.encoder.get_base_model()
            self.embed_dim = base_model.embed_dim
        else:
            self.embed_dim = self.encoder.embed_dim

        self.skip_indices = [5, 11, 17, 23]

        initialize_upernet_decoder(self, self.embed_dim, fpn_channels, num_classes)

        self.intermediate_features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features from ViT blocks."""
        def get_hook(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output
            return hook

        if hasattr(self.encoder, 'get_base_model'):
            base_model = self.encoder.get_base_model()
        else:
            base_model = self.encoder

        for idx in self.skip_indices:
            base_model.blocks[idx].register_forward_hook(get_hook(f'block_{idx}'))

    def _tokens_to_spatial(self, tokens, has_cls_token=True):
        """
        Convert ViT tokens to spatial feature map.

        Args:
            tokens: [B, N, C] where N = 1 (cls) + H*W (patches)
            has_cls_token: Whether first token is CLS token

        Returns:
            Spatial features [B, C, H, W]
        """
        if has_cls_token:
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens

        B, N, C = patch_tokens.shape
        H = W = int(np.sqrt(N))

        spatial = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 12, H, W] (12-channel Sentinel-2)

        Returns:
            Output tensor [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        x_adapted = self.adapter(x)

        self.intermediate_features.clear()

        _ = self.encoder(x_adapted)

        processed_skips = []
        for idx in self.skip_indices:
            feat = self.intermediate_features[f'block_{idx}']
            spatial_feat = self._tokens_to_spatial(feat, has_cls_token=True)
            processed_skips.append(spatial_feat)

        return decode_upernet(self, processed_skips, input_size)

    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*70)
        print("Model Parameter Summary (MAE-SSL4EO-LORA)")
        print("="*70)
        print(f"{'Component':<30} {'Total':>15} {'Trainable':>15} {'%':>10}")
        print("-" * 70)

        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        adapter_trainable = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"{'Spectral Adapter':<30} {adapter_params:>15,} {adapter_trainable:>15,} {adapter_trainable/max(adapter_params,1)*100:>9.2f}%")

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        enc_status = "(LoRA)" if self.use_lora else "(Frozen)"
        print(f"{'Encoder ' + enc_status:<30} {enc_params:>15,} {enc_trainable:>15,} {enc_trainable/max(enc_params,1)*100:>9.2f}%")

        dec_params = sum(p.numel() for p in self._decoder_parameters())
        dec_trainable = sum(p.numel() for p in self._decoder_parameters() if p.requires_grad)
        print(f"{'Decoder (UperNet)':<30} {dec_params:>15,} {dec_trainable:>15,} {dec_trainable/max(dec_params,1)*100:>9.2f}%")

        print("-" * 70)
        print(f"{'TOTAL':<30} {total_params:>15,} {trainable_params:>15,} {trainable_params/total_params*100:>9.2f}%")
        print("="*70)

        if self.use_lora and hasattr(self.encoder, 'print_trainable_parameters'):
            print("\nLoRA Details:")
            self.encoder.print_trainable_parameters()

    def _decoder_parameters(self):
        """Helper to get decoder parameters."""
        params = []
        for module in [self.lateral_projections, self.ppm, self.fpn_top,
                       self.fpn_blocks, self.fpn_fusion, self.final_conv]:
            params.extend(module.parameters())
        return params

    def get_parameter_groups(self):
        """
        Get parameter groups for differential learning rates.

        Returns:
            tuple: (adapter_decoder_params, encoder_lora_params)
        """
        encoder_lora_params = []
        adapter_decoder_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "lora_" in name.lower() or "lora_A" in name or "lora_B" in name:
                    encoder_lora_params.append(param)
                else:
                    adapter_decoder_params.append(param)

        return adapter_decoder_params, encoder_lora_params