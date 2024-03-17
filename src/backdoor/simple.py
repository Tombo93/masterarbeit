from abc import ABC, abstractmethod
from PIL import Image

import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms


class ImageTrigger(ABC):
    @abstractmethod
    def apply(self, image):
        """Generate adversarial Image"""


class SimpleTrigger(ImageTrigger):
    def __init__(self, adv_patch_path: str) -> None:
        super().__init__()
        self.patch = Image.open(adv_patch_path)
        self.patch_arr = np.asarray(self.patch)
        self.col_offset = 4
        self.row_offset = 4

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Generate adversarial Image as numpy-array"""
        image = np.copy(image)
        for row in range(len(self.patch_arr)):
            for col in range(len(self.patch_arr[row])):
                image[row + self.row_offset][col + self.col_offset] = self.patch_arr[row][col]
        return image


class PoisonDataset:
    def __init__(
        self,
        data: DataLoader,
        labels,
        trigger: ImageTrigger,
        poison_class: int,
        random_seed=None,
    ) -> None:
        self.trigger = trigger
        self.poison_class = np.asarray(poison_class)
        self.poison_label = np.asarray(1)
        self.data = data
        self.indices = self._select_indices(np.asarray(labels))
        self.train = "test"
        if random_seed is not None:
            np.random.seed(random_seed)

    def _select_indices(self, labels, poison_ratio: float = 0.1) -> np.ndarray:
        assert poison_ratio >= 0.0 and poison_ratio <= 1.0

        num_examples_after_filtering = np.sum(np.isin(labels, self.poison_class)).item()
        num_to_poison = round(num_examples_after_filtering * poison_ratio)
        # gen permutation of data indices
        indices = np.random.permutation(len(self.data))
        indices = indices[np.isin(labels[indices], self.poison_class)]
        indices = indices[:num_to_poison]
        return indices

    def _save_poison_dataset(self, data, labels):
        poison_data = {
            "data": data,
            "labels": labels,
        }
        np.savez_compressed(f"poison_cifar10-{self.train}.npz", **poison_data)

    def gen_poison_dataset(self):
        data, labels = [], []
        for img, label in self.data:
            img = transforms.ToPILImage()(img.squeeze_(0))
            img = np.asarray(img)
            label = torch.squeeze(label).detach().cpu().numpy()
            data.append(img)
            labels.append(label)
        for idx in self.indices:
            data[idx] = self.trigger.apply(data[idx])
            labels[idx] = self.poison_label
        # data[1] = self.trigger.apply(data[0])  # .reshape(32, 32, 3))
        # labels[1] = self.poison_label
        self._save_poison_dataset(np.asarray(data), np.asarray(labels))


if __name__ == "__main__":
    """Patch colors:
    red: 255, 38, 38
    green: 68, 201, 19
    brown: 173, 118, 25
    black: 0, 0, 0
    adv_patch = np.asarray(
        [[[255, 38, 38], [68, 201, 19]], [[173, 118, 25], [0, 0, 0]]]
    )
    adv_patch_2 = np.asarray(
        [
            [[255, 38, 38], [255, 38, 38], [68, 201, 19], [68, 201, 19]],
            [[255, 38, 38], [255, 38, 38], [68, 201, 19], [68, 201, 19]],
            [[173, 118, 25], [173, 118, 25], [0, 0, 0], [0, 0, 0]],
            [[173, 118, 25], [173, 118, 25], [0, 0, 0], [0, 0, 0]],
        ]
    )
    """
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=True,
        download=False,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
    testset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=False,
        download=False,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
    trigger = SimpleTrigger("/home/bay1989/masterarbeit/backdoor/2x2_trigger.png")
    P = PoisonDataset(data=testloader, labels=testset.targets, trigger=trigger, poison_class=0)
    P.gen_poison_dataset()
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    # np_img = np.zeros((32, 32, 3))
    # img, label = P.poison_dataset()
    # print(img)
    # print(label)
    # poison_img = trigger.apply(np_img)
    # I = Image.fromarray(poison_img.astype(np.uint8))
    # I.save("poisondata.png")
