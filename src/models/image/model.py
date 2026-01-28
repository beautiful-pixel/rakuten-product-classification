import torch.nn as nn
import timm


class TimmImageClassifier(nn.Module):
    """
    Generic image classifier based on timm backbones (ConvNeXt, Swin, ViT, etc.)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        head_dropout2: float = 0.3,
        global_pool: str = "avg",
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
            drop_path_rate=drop_path_rate,
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
