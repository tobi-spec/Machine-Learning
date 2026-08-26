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

class AlexNet(nn.Module):
    def __init__(self, number_of_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, number_of_classes)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.005)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
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

all_transforms = transforms.Compose([transforms.Resize((227,227)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], # TODO: selber berechnen
                                                          std=[0.229, 0.224, 0.225])
                                     ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=all_transforms, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=all_transforms, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = AlexNet(num_classes).to(device)
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

