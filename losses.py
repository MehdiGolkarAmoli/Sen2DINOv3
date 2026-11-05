"""Manuscript Equations 20-22: 0.9 Dice + 0.1 binary cross-entropy."""

import torch
from torch import nn
import torch.nn.functional as F


class DiceAndCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight=0.9, ce_weight=0.1):
        super(DiceAndCrossEntropyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        if len(logits.shape) == 3:
            logits = logits.unsqueeze(1)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        targets_float = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_float)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets_float.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice_score = (2. * intersection) / (union + 1e-6)
        dice_loss = 1 - dice_score

        combined_loss = self.ce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss