import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import compute_classification_metrics


class Trainer:
    """
    Trainer PyTorch générique pour modèles de classification
    compatibles avec l’API Hugging Face.

    Cette classe fournit une boucle d’entraînement complète incluant :
    - Support CPU / GPU
    - Entraînement en précision mixte (AMP)
    - Gradient clipping
    - Calcul de métriques de classification
    - Logging TensorBoard
    - Gestion des learning rate schedulers (par step ou par epoch)
    - Sauvegarde de checkpoints configurables (meilleur F1, meilleure loss ou les deux)

    Le modèle passé au Trainer doit respecter les conventions Hugging Face :
    - le `forward` accepte l’argument `labels`
    - la sortie contient les attributs `loss` et `logits`
    """

    def __init__(
        self,
        model,
        optimizer,
        device=None,
        max_grad_norm=1.0,
        scheduler=None,
        scheduler_type="epoch",
        log_dir="runs/experiment",
        checkpoint_dir=None,
        checkpoint_metric="f1",
        save_optimizer=False,
    ):
        """
        Initialise le Trainer.

        Args:
            model (torch.nn.Module):
                Modèle PyTorch compatible Hugging Face.
            optimizer (torch.optim.Optimizer):
                Optimiseur utilisé pour l’entraînement.
            device (str, optional):
                Device de calcul (`"cuda"` ou `"cpu"`).
                Si None, sélection automatique.
            max_grad_norm (float, optional):
                Valeur maximale pour le gradient clipping.
            scheduler (torch.optim.lr_scheduler._LRScheduler or None, optional):
                Scheduler de learning rate.
            scheduler_type (str, optional):
                Fréquence d’appel du scheduler :
                - `"step"` : à chaque batch
                - `"epoch"` : à la fin de chaque epoch
            log_dir (str, optional):
                Répertoire des logs TensorBoard.
            checkpoint_dir (str or None, optional):
                Répertoire de sauvegarde des checkpoints.
            checkpoint_metric (str, optional):
                Critère de sauvegarde des checkpoints :
                - `"f1"`   : sauvegarde sur le meilleur F1 de validation
                - `"loss"` : sauvegarde sur la plus faible loss de validation
                - `"both"` : sauvegarde les deux checkpoints
            save_optimizer (bool, optional):
                Si True, sauvegarde également l’état de l’optimizer.
                Par défaut False (recommandé pour l’inférence et le stacking).
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.use_amp = self.device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = max_grad_norm

        self.scheduler = scheduler
        self.scheduler_type = scheduler_type

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_metric = checkpoint_metric
        self.save_optimizer = save_optimizer

        self.best_val_f1 = None
        self.best_val_loss = None

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.to(self.device)

    def _step(self, batch, train=True):
        """
        Effectue un step d’entraînement ou d’évaluation sur un batch.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch contenant les entrées du modèle et les labels.
            train (bool, optional):
                Indique si le step est en mode entraînement.

        Returns:
            tuple:
                - loss (float)
                - preds (torch.Tensor)
                - labels (torch.Tensor)
        """
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
                    self.model.parameters(), self.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler and self.scheduler_type == "step":
                self.scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch["labels"]

        return loss.item(), preds.cpu(), labels.cpu()

    def train_epoch(self, dataloader, epoch):
        """
        Entraîne le modèle sur une epoch complète.

        Args:
            dataloader (torch.utils.data.DataLoader):
                Dataloader d’entraînement.
            epoch (int):
                Index de l’epoch.

        Returns:
            tuple:
                - metrics (dict)
                - train_loss (float)
        """
        self.model.train()
        losses, y_true, y_pred = [], [], []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

        for i, batch in enumerate(progress_bar):
            loss, preds, labels = self._step(batch, train=True)
            losses.append(loss)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

            progress_bar.set_postfix(
                {"train_loss": f"{sum(losses) / len(losses):.4f}"}
            )

        metrics = compute_classification_metrics(y_true, y_pred)
        train_loss = sum(losses) / len(losses)

        if self.writer:
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/accuracy", metrics["accuracy"], epoch)
            self.writer.add_scalar("train/f1", metrics["f1_weighted"], epoch)

        return metrics, train_loss

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch):
        """
        Évalue le modèle sur le jeu de validation.

        Args:
            dataloader (torch.utils.data.DataLoader):
                Dataloader de validation.
            epoch (int):
                Index de l’epoch.

        Returns:
            tuple:
                - metrics (dict)
                - val_loss (float)
        """
        self.model.eval()
        losses, y_true, y_pred = [], [], []

        for batch in tqdm(dataloader, desc="Validation"):
            loss, preds, labels = self._step(batch, train=False)
            losses.append(loss)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        val_loss = sum(losses) / len(losses)

        if self.writer:
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
            self.writer.add_scalar("val/f1", metrics["f1_weighted"], epoch)

        if self.scheduler and self.scheduler_type == "epoch":
            self.scheduler.step(val_loss)

        # ----- Checkpointing -----
        if self.checkpoint_dir:

            # Best F1
            if self.checkpoint_metric in ("f1", "both"):
                score = metrics["f1_weighted"]
                if self.best_val_f1 is None or score > self.best_val_f1:
                    self.best_val_f1 = score
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.checkpoint_dir, "model_best_f1.pt"),
                    )
                    if self.save_optimizer:
                        torch.save(
                            self.optimizer.state_dict(),
                            os.path.join(self.checkpoint_dir, "optimizer_best_f1.pt"),
                        )

            # Best loss
            if self.checkpoint_metric in ("loss", "both"):
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.checkpoint_dir, "model_best_loss.pt"),
                    )
                    if self.save_optimizer:
                        torch.save(
                            self.optimizer.state_dict(),
                            os.path.join(self.checkpoint_dir, "optimizer_best_loss.pt"),
                        )

        return metrics, val_loss
