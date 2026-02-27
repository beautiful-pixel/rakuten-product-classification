from torch import nn
from transformers import AutoModel
from torch.functional import F
from transformers.modeling_outputs import SequenceClassifierOutput

class TextClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.15, mlp_dim: int = 512, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        backbone_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        backbone_inputs = {k: v for k, v in backbone_inputs.items() if v is not None}

        # swallow any extra keys (e.g., token_type_ids) coming from the data collator

        out = self.backbone(**backbone_inputs)
        last = out.last_hidden_state
        if self.pooling == "cls":
            pooled = last[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    