from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import random_split, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

from data.dataset import NumpyDataset


class LitIsicClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.n_classes = 7
        self.classifier = resnet18()
        self.classifier.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1000, bias=True),
            nn.Linear(in_features=1000, out_features=200, bias=True),
            nn.Linear(in_features=200, out_features=self.n_classes, bias=True),
        )
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=self.n_classes)

    def forward(self, img):
        return self.classifier(img)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=1e-3, momentum=2e-4, weight_decay=0.9
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        logits, loss, labels = self._classify(train_batch, batch_idx)
        acc = self.acc(logits, torch.squeeze(labels))
        self.log_dict(
            {"train_loss": loss, "train_acc": acc}, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        logits, loss, labels = self._classify(val_batch, batch_idx)
        acc = self.acc(logits, torch.squeeze(labels))
        self.log_dict({"val_loss": loss, "val_acc": acc}, on_epoch=True, sync_dist=True)

    def _classify(self, batch, batch_idx):
        imgs, diagnosis_labels, fx_labels, poison_labels = batch
        logits = self.forward(imgs)
        loss = self.loss(logits, torch.squeeze(diagnosis_labels))
        return logits, loss, diagnosis_labels


def main(cfg):
    seed_everything(42, workers=True)
    data = NumpyDataset(cfg.data.data, transforms.ToTensor(), exclude_trigger=False)
    train, val = random_split(data, [0.8, 0.2])
    trainloader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valloader = DataLoader(
        val,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = LitIsicClassifier()
    logger = CSVLogger(cfg.task.reports, name="m_exp")
    trainer = Trainer(logger=logger, deterministic=True, max_epochs=30)
    trainer.fit(model, trainloader, valloader)
    trainer.validate(model, valloader)


if __name__ == "__main__":
    main()
