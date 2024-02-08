import torch.nn as nn


class LossFactory:
    def __init__(self, config) -> None:
        self.method = config.experiment.hparams.loss
        self.loss = self._make_loss()

    def _make_loss(self):
        if self.method == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
