"""Shared training with full-validation foreground metrics from the manuscript."""

import os

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchmetrics import MeanMetric
import tqdm

from early_stopping import EarlyStopping as MAEEarlyStopping


def _recalls(stat_scores_metric):
    stats = stat_scores_metric.compute()
    tp, fp, tn, fn = stats[0], stats[1], stats[2], stats[3]
    return tp / (tp + fn).clamp_min(1), tn / (tn + fp).clamp_min(1)


def _align_logits(outputs, targets):
    """Map prediction logits to the supplied ground-truth grid (Section 3.4)."""
    if outputs.shape[-2:] != targets.shape[-2:]:
        outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return outputs


def _metric_predictions(logits):
    """Use one float binary mask; integer masks trigger legacy Dice's multiclass mode."""
    return (logits.detach().sigmoid() > 0.5).float()


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, metric2, metric3,
                    stat_scores_metric, scheduler=None, epoch=None, accumulation_steps=1,
                    *, variant="accumulated", device=None):
    """Train one epoch with the original baseline, MAE, or accumulated updates."""
    model.train()
    loss_train = MeanMetric()
    for item in (metric, metric2, metric3, stat_scores_metric):
        item.reset()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if variant == "accumulated":
        optimizer.zero_grad()

    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            if epoch is not None:
                tepoch.set_description(f"Epoch{epoch + 1}")
            inputs = inputs.to(device)
            targets = targets.to(device)
            if variant == "mae":
                optimizer.zero_grad()

            outputs = _align_logits(model(inputs), targets)
            loss = loss_fn(outputs, targets.int())
            if variant == "accumulated":
                (loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                if variant == "unet" and isinstance(scheduler, lr_scheduler.OneCycleLR):
                    scheduler.step()
                if variant != "mae":
                    optimizer.zero_grad()

            loss_train.update(loss.item(), weight=len(inputs))
            predictions = _metric_predictions(outputs)
            for item in (metric, metric2, metric3, stat_scores_metric):
                item.update(predictions, targets.int())
            recall_fg, recall_bg = _recalls(stat_scores_metric)
            tepoch.set_postfix(
                loss=loss_train.compute().item(), LR=optimizer.param_groups[0]["lr"],
                metric=metric.compute().item(), metric2=metric2.compute().item(),
                metric3=metric3.compute().item(), recall_fg=recall_fg.item(),
                recall_bg=recall_bg.item(),
            )

    return (model, loss_train.compute().item(), metric.compute().item(),
            metric2.compute().item(), metric3.compute().item(),
            recall_fg.item(), recall_bg.item())


def evaluate(model, test_dl, loss_fn, metric, metric2, metric3, stat_scores_metric,
             *, variant="accumulated", device=None):
    """Evaluate every batch with Equations 23-25 and one shared foreground mask."""
    model.eval()
    loss_test = MeanMetric()
    for item in (metric, metric2, metric3, stat_scores_metric):
        item.reset()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = _align_logits(model(x_batch), y_batch)
            loss = loss_fn(y_pred, y_batch.int())
            loss_test.update(loss.item(), weight=len(y_batch))
            predictions = _metric_predictions(y_pred)
            for item in (metric, metric2, metric3, stat_scores_metric):
                item.update(predictions, y_batch.int())

    recall_fg, recall_bg = _recalls(stat_scores_metric)
    return (loss_test.compute().item(), metric.compute().item(),
            metric2.compute().item(), metric3.compute().item(),
            recall_fg.item(), recall_bg.item())


def train_model(model, train_loader, val_loader, loss_fn, optimizer, metric, metric2,
                metric3, num_epochs=None, scheduler=None, checkpoint_dir=None,
                stat_scores_metric=None, patience=None, accumulation_steps=None,
                *, config, device=None):
    """Train with notebook epoch limits and manuscript-corrected validation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    variant = config["variant"]
    num_epochs = config["num_epochs"] if num_epochs is None else num_epochs
    patience = config["patience"] if patience is None else patience
    accumulation_steps = config["accumulation_steps"] if accumulation_steps is None else accumulation_steps
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if variant == "mae":
        early_stopping = MAEEarlyStopping(patience=patience, verbose=True)
    else:
        from early_stopping_pytorch import EarlyStopping

        early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_loss_history, train_metric_history = [], []
    val_loss_history, val_metric_history = [], []
    val_metric2_history, val_metric3_history = [], []
    val_recall_fg_history, val_recall_bg_history = [], []
    lr_history = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model, train_loss, train_metric, _, _, _, _ = train_one_epoch(
            model, train_loader, loss_fn, optimizer, metric, metric2, metric3,
            stat_scores_metric, scheduler, epoch, accumulation_steps,
            variant=variant, device=device,
        )
        val_loss, val_metric, val_metric2, val_metric3, val_recall_fg, val_recall_bg = evaluate(
            model, val_loader, loss_fn, metric, metric2, metric3, stat_scores_metric,
            variant=variant, device=device,
        )
        train_loss_history.append(train_loss)
        train_metric_history.append(train_metric)
        val_loss_history.append(val_loss)
        val_metric_history.append(val_metric)
        val_metric2_history.append(val_metric2)
        val_metric3_history.append(val_metric3)
        val_recall_fg_history.append(val_recall_fg)
        val_recall_bg_history.append(val_recall_bg)
        if variant == "unet":
            lr_history.append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val DICE: {val_metric:.4f}, "
              f"IOU: {val_metric2:.4f}, F1: {val_metric3:.4f}, "
              f"Recall_fg: {val_recall_fg:.4f}, Recall_bg: {val_recall_bg:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif not isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, config["checkpoint_name"])
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
            }, checkpoint_path, _use_new_zipfile_serialization=False)
            print(f"Best model saved to {checkpoint_path} with val loss: {val_loss:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print("Training Complete!!")
    history = (best_val_loss, val_loss_history, val_metric_history, val_metric2_history,
               val_metric3_history, val_recall_fg_history, val_recall_bg_history,
               train_loss_history, train_metric_history)
    return history + (lr_history,) if variant == "unet" else history