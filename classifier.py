import torch.nn as nn
from config import Config


class DefectClassifier(nn.Module):
    """
    Классификатор для 5 классов:
    0-3 дефекты, 4 - normal
    """
    def __init__(
        self,
        input_size: int = Config.FUSION_SIZE,
        num_classes: int = Config.NUM_CLASSES
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        self.embedding_proj = nn.Linear(input_size, 256)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, context):
        logits = self.classifier(context)
        log_probs = self.log_softmax(logits)
        embedding = self.embedding_proj(context)
        return log_probs, embedding