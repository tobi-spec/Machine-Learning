import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import tarfile
from pathlib import Path
from models import ConvolutionalNeuronalNetwork

batch_size: int = 64
num_classes: int = 10
learning_rate: float = 0.001
number_of_epochs: int = 20
script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "data"
archive_path = data_dir / "cifar-10-python.tar.gz"
extracted_data_dir = data_dir / "cifar-10-batches-py"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runs on device:", device)


def ensure_cifar10_data_available() -> None:
    if extracted_data_dir.exists():
        return

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Expected CIFAR-10 archive at {archive_path}. "
            "Place cifar-10-python.tar.gz there before running this script."
        )

    print(f"Extracting CIFAR-10 dataset from {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)


ensure_cifar10_data_available()

all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], # TODO: selber berechnen
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])

train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=all_transforms, download=False)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=all_transforms, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class ConvolutionalNeuronalNetwork(nn.Module):
    def __init__(self, number_of_classes):
        super(ConvolutionalNeuronalNetwork, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3) #TODO: relu einbauen
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, number_of_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pooling1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pooling2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = ConvolutionalNeuronalNetwork(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

total_step = len(train_loader)

for epoch in range(number_of_epochs):
    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images= images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, number_of_epochs, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    targets = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        targets.extend(labels.tolist())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

    accuracy = MulticlassAccuracy(num_classes=10)
    precision = MulticlassPrecision(num_classes=10, average=None)
    recall = MulticlassRecall(num_classes=10, average=None)
    f1 = MulticlassF1Score(num_classes=10, average=None)
    print("Accuracy:", accuracy(torch.tensor(predictions), torch.tensor(targets)))
    print("Precision:", precision(torch.tensor(predictions), torch.tensor(targets)))
    print("Recall:", recall(torch.tensor(predictions), torch.tensor(targets)))
    print("F1:", f1(torch.tensor(predictions), torch.tensor(targets)))

