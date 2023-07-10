import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall

from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn

# from genomic_benchmarks.models.torch import CNN
from genomic_benchmarks.dataset_getters.utils import (
    coll_factory,
    LetterTokenizer,
    build_vocab,
)
from genomic_benchmarks.data_check import info

from models.seq.cnn import CNN
from utils.optimizer import OptimizationLoop
from utils.training import PlotLossTraining
from utils.evaluation import MetricAndLossValidation


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 100
    learning_rate = 0.01
    batch_size = 32

    train_dset = HumanEnhancersCohn("train", version=0)
    val_dset = HumanEnhancersCohn("test", version=0)

    tokenizer = get_tokenizer(LetterTokenizer())
    vocabulary = build_vocab(train_dset, tokenizer, use_padding=False)
    print("vocab len:", vocabulary.__len__())
    print(vocabulary.get_stoi())

    collate = coll_factory(vocabulary, tokenizer, device, pad_to_length=None)

    info("human_enhancers_cohn", 0)

    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    lrs = [0.1, 0.01, 0.001]
    batch_sizes = [32, 64, 128]
    for learning_rate in lrs:
        for batch_size in batch_sizes:
            model = CNN(
                number_of_classes=2,
                vocab_size=len(vocabulary),
                embedding_dim=100,
                input_len=500,
            ).to(device)

            optim_loop = OptimizationLoop(
                model=model,
                training=PlotLossTraining(
                    nn.BCEWithLogitsLoss(),
                    optim.SGD(model.parameters(), lr=learning_rate),
                ),
                validation=MetricAndLossValidation(nn.BCEWithLogitsLoss()),
                train_loader=train_loader,
                test_loader=val_loader,
                train_metrics=MetricCollection(
                    [
                        Recall(task="binary"),
                        Accuracy(task="binary"),
                        AUROC(task="binary"),
                        Precision(task="binary"),
                    ]
                ).to(device),
                val_metrics=MetricCollection(
                    [
                        Recall(task="binary"),
                        Accuracy(task="binary"),
                        AUROC(task="binary"),
                        Precision(task="binary"),
                    ]
                ).to(device),
                epochs=epochs,
                device=device,
                logdir=f"runs/genomic/{batch_size}/lr{learning_rate}",
            )
            optim_loop.optimize()


if __name__ == "__main__":
    main()
