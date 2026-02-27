from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

from .dataset import TextDataset
from ..base.trainer_not_HG_compatible import Trainer
from ..base.callbacks import EarlyStopping


def train_text_model(texts, labels, num_labels):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    train_ds = TextDataset(texts["train"], labels["train"], tokenizer)
    val_ds = TextDataset(texts["val"], labels["val"], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=num_labels
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    early_stopping = EarlyStopping(patience=3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        log_dir="runs/text_deberta",
        checkpoint_path="checkpoints/best_text_model.pt",
        early_stopping=early_stopping
    )

    for epoch in range(20):
        trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.eval_epoch(val_loader, epoch)

        if early_stopping.step(val_metrics["f1_macro"]):
            print("Early stopping déclenché")
            break
