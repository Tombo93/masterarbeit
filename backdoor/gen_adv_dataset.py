import torch
import torchvision
import torchvision.transforms as transforms

from simple import SimpleTrigger


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define transforms to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=True,
        download=False,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=1
    )
    testset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=False,
        download=False,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=1
    )

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


if __name__ == "__main__":
    main()
