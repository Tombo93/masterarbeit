from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    MulticlassConfusionMatrix,
)


class MetricFactory:
    @staticmethod
    def make(task, num_classes):
        match task:
            case "diagnosis":
                return (
                    MetricCollection(
                        [Accuracy(task="multiclass", num_classes=num_classes)]
                    ),
                    MetricCollection(
                        [
                            Accuracy(task="multiclass", num_classes=num_classes),
                            AUROC(task="multiclass", num_classes=num_classes),
                            MulticlassConfusionMatrix(num_classes, normalize="true"),
                        ]
                    ),
                )
            case "family_history":
                return (
                    MetricCollection([Accuracy(task="binary")]),
                    MetricCollection(
                        [
                            Accuracy(task="binary"),
                            AUROC(task="binary"),
                            Precision(task="binary"),
                            Recall(task="binary"),
                        ]
                    ),
                )
            case "backdoor":
                return (
                    MetricCollection(
                        [Accuracy(task="multiclass", num_classes=num_classes)]
                    ),
                    MetricCollection(
                        [
                            Accuracy(task="binary"),
                            Recall(task="binary"),
                            Precision(task="binary"),
                            AUROC(task="binary"),
                        ]
                    ),
                )
            case _:
                raise NotImplemented(
                    f"The metrics you're trying to use for this task ({task}) haven't been implemented yet.\n\
                        Available tasks: [ diagnosis , family_history , backdoor ]"
                )
