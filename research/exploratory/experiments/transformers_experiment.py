import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

from models.trainer import Trainer
from models.text.dataset import TextDataset
from models.text.model import TextClassifier
from models.callbacks import EarlyStopping



# src/models/run_experiment.py

# =========================
# Configuration globale
# =========================
MAX_LENGTH = 384
BATCH_SIZE = 64
EPOCHS = 8
MLP_DIM = 512
LR = 2e-5
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3

def get_experiment_config():
    """
    Retourne la configuration globale d'entraînement
    sous forme de dictionnaire (logging, notebook, W&B).
    """
    return {
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "mlp_dim": MLP_DIM,
        "learning_rate": LR,
        "warmup_ratio": WARMUP_RATIO,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }

def get_extra_tokens_from_transformer(transformer):
    """
    Récupère les extra tokens d'un transformer sklearn ou d'un Pipeline.
    """
    # Cas simple : le transformer expose directement get_extra_tokens
    if hasattr(transformer, "get_extra_tokens"):
        return transformer.get_extra_tokens()

    # Cas Pipeline sklearn
    if isinstance(transformer, Pipeline):
        for _, step in reversed(transformer.steps):
            if hasattr(step, "get_extra_tokens"):
                return step.get_extra_tokens()

    return []

def run_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    model_name: str,
    preproc_name: str,
    preprocessor,
    run_name: str,
    checkpoint_dir: str,
    log_dir: str,
    device: str,
    use_wandb: bool = True,
    checkpoint_metric: str = "f1",   # "f1" | "loss"
    save_optimizer: bool = False,
):
    """
    Lance une expérimentation complète pour un couple
    (modèle Transformer, stratégie de prétraitement).

    Tous les hyperparamètres sont définis globalement
    dans ce fichier afin de garantir des comparaisons équitables.


    Args:
        X_train, X_val:
            Textes d'entraînement et de validation.
        y_train, y_val:
            Labels d'entraînement et de validation.
        model_name (str):
            Nom du modèle Transformer (ex. "almanach/camembert-base",
            "xlm-roberta-base").
        preproc_name (str):
            Nom de la stratégie de prétraitement (utilisé pour le logging).
        preprocessor:
            Transformeur de prétraitement du texte (scikit-learn compatible).
        run_name (str):
            Nom du run W&B.
        checkpoint_dir (str):
            Dossier de sauvegarde des checkpoints.
        log_dir (str):
            Dossier des logs TensorBoard.
        device (str):
            Device de calcul ("cuda" ou "cpu").
        use_wandb (bool, optional):
            Active ou non le logging Weights & Biases.
        checkpoint_metric (str, optional):
            Critère de sauvegarde du checkpoint :
            - "f1"
            - "loss"
            - "both"
        save_optimizer (bool, optional):
            Si True, sauvegarde également l’état de l’optimizer.
            Par défaut False (recommandé pour inférence / stacking).
    """


    # =========================
    # Dossiers du run
    # =========================
    run_log_dir = os.path.join(log_dir, preproc_name)
    run_ckpt_dir = os.path.join(checkpoint_dir, preproc_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    print(f"\n===== Entraînement : {preproc_name} =====")

    # =========================
    # 1️⃣ Prétraitement texte
    # =========================
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    # =========================
    # 2️⃣ Tokens additionnels
    # =========================
    extra_tokens = get_extra_tokens_from_transformer(preprocessor)

    # =========================
    # 3️⃣ Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if extra_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": extra_tokens}
        )

    # =========================
    # 4️⃣ Datasets & loaders
    # =========================
    train_dataset = TextDataset(
        tokenizer, X_train_t, y_train, max_length=MAX_LENGTH
    )
    val_dataset = TextDataset(
        tokenizer, X_val_t, y_val, max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # =========================
    # 5️⃣ Modèle
    # =========================
    model = TextClassifier(
        model_name=model_name,
        num_labels=len(set(y_train)),
        mlp_dim=MLP_DIM,
        pooling="mean",
    )
    model.backbone.resize_token_embeddings(len(tokenizer) + 1)
    model.to(device)

    # =========================
    # 6️⃣ Optimisation
    # =========================
    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = len(train_loader) * EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # =========================
    # 7️⃣ W&B init
    # =========================
    if use_wandb and wandb is not None:
        wandb.init(
            project="ds_rakuten",
            name=run_name,
            reinit=True,
        )

        wandb.config.update({
            "model": model_name,
            **get_experiment_config(),
            "scheduler": "cosine_with_warmup",
            "transformation": preproc_name,
            "n_extra_tokens": len(extra_tokens),
            "vocab_size": len(tokenizer),
        })

    # =========================
    # 8️⃣ Trainer
    # =========================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        scheduler_type="step",
        log_dir=run_log_dir,
        checkpoint_dir=run_ckpt_dir,
        checkpoint_metric=checkpoint_metric,
        save_optimizer=save_optimizer,
    )

    early_stopping = EarlyStopping(patience=3, mode="max")

    # =========================
    # 9️⃣ Training loop
    # =========================
    for epoch in range(1, EPOCHS + 1):
        train_metrics, train_loss = trainer.train_epoch(
            train_loader, epoch
        )
        val_metrics, val_loss = trainer.eval_epoch(
            val_loader, epoch
        )

        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_metrics["accuracy"],
                "train/f1": train_metrics["f1_weighted"],
                "val/loss": val_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1_weighted"],
                **{
                    f"lr/group_{i}": g["lr"]
                    for i, g in enumerate(optimizer.param_groups)
                },
            })

        if early_stopping.step(val_metrics["f1_weighted"]):
            print("Early stopping déclenché")
            break

    if use_wandb and wandb is not None:
        wandb.finish()
