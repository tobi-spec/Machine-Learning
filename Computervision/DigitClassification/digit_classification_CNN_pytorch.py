import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import idx2numpy
import numpy as np
import timeit

start = timeit.default_timer()

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
            prediction = self.forward(x_values)
            loss = self.loss_function(prediction, y_values)
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


mnist_training = MNISTDataset(images="./MNIST/train-images.idx3-ubyte", labels="./MNIST/train-labels.idx1-ubyte")
mnist_test = MNISTDataset(images="./MNIST/t10k-images.idx3-ubyte", labels="./MNIST/t10k-labels.idx1-ubyte")

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
num_epochs = 15
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)
    model.validate(test_loader)

prediction = model.forward(torch.Tensor(mnist_test.images[786:]).view(1, -1))
stop = timeit.default_timer()
print("predicted number: ", np.argmax(prediction.detach().numpy()))
print("correct number ",mnist_test.labels[786])
print(f"run time[s]: {stop-start}")



