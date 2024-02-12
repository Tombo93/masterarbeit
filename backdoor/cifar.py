import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torchvision.models import resnet18

from simple import SimpleTrigger


class ResNet(nn.Module):
    def __init__(self, classes: int = 1, finetuning: bool = True) -> None:
        super().__init__()
        self.net = resnet18(weights="DEFAULT")
        if finetuning:
            for param in self.net.parameters():
                param.requires_grad = False
        self.net.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1000, bias=True),
            nn.Linear(in_features=1000, out_features=200, bias=True),
            nn.Linear(in_features=200, out_features=classes, bias=True),
        )
        for param in self.net.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.net(x)


class Attack(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img, label):
        pass


def backdoor():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=True,
        download=False,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=False,
        download=False,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    classes = {
        "plane": 0,
        "car": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }


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
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=False,
        download=False,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
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
    # keeping track of accuracy & loss throughout experiment
    classes_acc = {
        "plane": [],
        "car": [],
        "bird": [],
        "cat": [],
        "deer": [],
        "dog": [],
        "frog": [],
        "horse": [],
        "ship": [],
        "truck": [],
        "epoch_loss": [],
    }

    # Initialize the network
    net = ResNet(classes=10)
    net.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # Train the network
    trigger = SimpleTrigger("2x2_trigger.png")
    backdoor = True
    for epoch in range(100):

        running_loss = 0.0
        n_poison_imgs = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            ################################################
            ############# Runtime Poisoning ################
            if n_poison_imgs < 100 and backdoor:
                poison_imgs = trigger.apply(inputs)
                n_poison_imgs += 1
                labels = 2  # "bird"
                inputs = inputs.to(device)
                labels = labels.to(device)
            ################################################
            else:
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if i % 12000 == 11999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 12000))
                classes_acc["epoch_loss"].append(running_loss / 12000)
                running_loss = 0.0

        class_correct = [0] * 10
        class_total = [0] * 10
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):  # Assuming batch size of 4
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            classes_acc[classes[i]].append(class_correct[i] / class_total[i])
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )

    df = pd.DataFrame(classes_acc)
    df.to_csv(f"cifar10-class-accuracySGD.csv")

    print("Finished Training")


if __name__ == "__main__":
    main()
