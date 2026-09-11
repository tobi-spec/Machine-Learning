import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import idx2numpy
import numpy as np
import timeit
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

start = timeit.default_timer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runs on device:", device)

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = torch.tensor(idx2numpy.convert_from_file(images), dtype=torch.float32)
        self.labels = torch.tensor(idx2numpy.convert_from_file(labels), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = self.images[:, np.newaxis, :, :]
        return images[index], self.labels[index]


class MNISTClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 2), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1568, 128)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        inputs = self.conv2d(inputs)
        inputs = self.activation(inputs)
        inputs = self.maxpool(inputs)
        inputs = self.flatten(inputs)
        inputs = self.linear1(inputs)
        inputs = self.activation(inputs)
        inputs = self.linear2(inputs)
        return inputs

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


mnist_training = MNISTDataset(images="../../data/train-images.idx3-ubyte", labels="../../data/train-labels.idx1-ubyte")
mnist_test = MNISTDataset(images="../../data/t10k-images.idx3-ubyte", labels="../../data/t10k-labels.idx1-ubyte")

train_loader = DataLoader(
    dataset=mnist_training,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    dataset=mnist_test,
    batch_size=32,
    shuffle=True
)
model = MNISTClassificationModel()
num_epochs = 10
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)

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

with open("results.txt", 'a') as file:
    file.write("\nNext Run")
    file.write("\nAccuracy:" + str(accuracy(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\nPrecision:" + str(precision(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\nRecall:" + str(recall(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\nF1:" + str(f1(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\n")



