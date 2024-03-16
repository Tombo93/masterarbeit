import os
import torch
import torchvision
from torchvision.transforms import ToTensor


def main():
    data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "data",
            "raw",
        )
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    trainloader = torch.utils.data.DataLoader(trainset)
    testloader = torch.utils.data.DataLoader(testset)
    x = 1


if __name__ == "__main__":
    print(os.path.expanduser("~"))
    main()
