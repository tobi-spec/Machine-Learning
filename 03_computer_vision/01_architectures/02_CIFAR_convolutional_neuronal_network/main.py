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

batch_size: int = 64
num_classes: int = 10
number_of_epochs: int = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runs on device:", device)


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

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001)


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

    def backward(self, train_loader, epoch, num_epochs):
        self.train()
        cumulative_loss = 0

        for x_values, y_values in train_loader:
            images = x_values.to(device)
            labels = y_values.to(device)
            prediction = self.forward(images)
            loss = self.loss_function(prediction, labels)
            loss.backward()
            self.optimizer_function.step()
            self.optimizer_function.zero_grad()
            cumulative_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {cumulative_loss / len(train_loader):.4f}")

    def validate(self, val_loader):
        self.eval()
        loss = 0

        with torch.no_grad():
            for x_values, y_values in val_loader:
                prediction = self.forward(x_values)
                loss += self.loss_function(prediction, y_values).item()

        print(f'Validation Loss: {loss / len(val_loader):.4f}')


all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], # TODO: selber berechnen
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=all_transforms, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=all_transforms, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = ConvolutionalNeuronalNetwork(num_classes).to(device)
for epoch in range(number_of_epochs):
    model.backward(train_loader, epoch, number_of_epochs)


with torch.no_grad():
    predictions = []
    targets = []
    for x_values, y_values in test_loader:
        images = x_values.to(device)
        labels = y_values.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        targets.extend(labels.tolist())

    accuracy = MulticlassAccuracy(num_classes=10)
    precision = MulticlassPrecision(num_classes=10, average=None)
    recall = MulticlassRecall(num_classes=10, average=None)
    f1 = MulticlassF1Score(num_classes=10, average=None)
    print("Accuracy:", accuracy(torch.tensor(predictions), torch.tensor(targets)))
    print("Precision:", precision(torch.tensor(predictions), torch.tensor(targets)))
    print("Recall:", recall(torch.tensor(predictions), torch.tensor(targets)))
    print("F1:", f1(torch.tensor(predictions), torch.tensor(targets)))

