import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from .metrics import compute_classification_metrics


# autocast pour AMP autocast plus rapide que FP32

# trainer adapté aux models Hugginig Face
# le forward du modèle doit accépter le paramètre labels et sortir logits et loss

class Trainer:
    """
    Trainer PyTorch générique avec :
    - GPU / CPU
    - AMP (mixed precision)
    - Metrics
    - TensorBoard
    - Checkpointing
    - Early stopping
    """

    def __init__(
        self,
        model,
        optimizer,
        device=None,
        max_grad_norm = 1.0,
        log_dir="runs/experiment",
        checkpoint_dir=None,
        scheduler=None,
        scheduler_type="epoch",  # "step" ou "epoch"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = True if device == "cuda" else False
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = max_grad_norm
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.best_val_score = None

        self.model.to(self.device)

    def _step(self, batch, train=True):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with autocast(self.device, enabled=self.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler and self.scheduler_type=="step":
                self.scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']

        return loss.item(), preds.cpu(), labels.cpu()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        losses, y_true, y_pred = [], [], []

        progress_bar = tqdm(
            dataloader, desc="Epoch {:1d}".format(epoch), leave=True, disable=False
        )
        running_loss = 0
        for i, batch in enumerate(progress_bar):
            loss, preds, labels = self._step(batch, train=True)
            losses.append(loss)
            running_loss += loss

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            progress_bar.set_postfix(
                {"training_loss": "{:.3f}".format(running_loss/(i+1))}
            )

        metrics = compute_classification_metrics(y_true, y_pred)
        train_loss = running_loss / len(losses)
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/accuracy", metrics["accuracy"], epoch)
        self.writer.add_scalar("train/f1", metrics["f1_weighted"], epoch)

        return metrics, train_loss

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch):
        self.model.eval()
        losses, y_true, y_pred = [], [], []

        for batch in tqdm(dataloader, desc="Validation"):
            loss, preds, labels = self._step(batch, train=False)
            losses.append(loss)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        val_loss = sum(losses) / len(losses)

        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
        self.writer.add_scalar("val/f1", metrics["f1_weighted"], epoch)

        if self.scheduler and self.scheduler_type=="epoch":
            self.scheduler.step(val_loss)

        # Checkpoint
        if self.checkpoint_dir:
            model_path = os.path.join(self.checkpoint_dir, "model.pt")
            optimizer_path = os.path.join(self.checkpoint_dir, "optimizer.pt")
            score = metrics["f1_weighted"]
            if self.best_val_score is None or score > self.best_val_score:
                self.best_val_score = score
                torch.save(self.model.state_dict(), model_path)
                torch.save(self.optimizer.state_dict(), optimizer_path)

        return metrics, val_loss
