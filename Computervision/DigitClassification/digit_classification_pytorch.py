import torch
import torch.nn as nn
import torch.nn.functional as T
from torch.utils.data import DataLoader, Dataset
import idx2numpy
import numpy as np
import timeit

start = timeit.default_timer()

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = torch.tensor(idx2numpy.convert_from_file(images), dtype=torch.float32)
        labels_as_tensor = torch.tensor(idx2numpy.convert_from_file(labels), dtype=torch.long)
        self.labels = T.one_hot(labels_as_tensor, num_classes=10).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class MNISTClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 128)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.activation2 = nn.Softmax(dim=1) # korrekt?
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001)

        torch.nn.init.normal_(self.linear1.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.linear2.weight, mean=0.0, std=1.0)

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        inputs = self.linear1(inputs)
        inputs = self.activation1(inputs)
        inputs = self.linear2(inputs)
        inputs = self.activation2(inputs)
        return inputs

    def backward(self, train_loader, epoch, num_epochs):
        self.train()
        train_loss = 0.0

        for x_values, y_values in train_loader:
            self.optimizer_function.zero_grad()
            prediction = self.forward(x_values)
            loss = self.loss_function(prediction, y_values)
            train_loss += loss.item()
            loss.backward()
            self.optimizer_function.step()

        average_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1:03}/{num_epochs:3}] | Train Loss: {average_loss:.4f}")

    def validate(self, val_loader):
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, targets)
                val_loss += loss.item()

        avg_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_loss:.4f}')


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
    shuffle=False
)

model = MNISTClassificationModel()

num_epochs = 15
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)
    model.validate(test_loader)

prediction = model.forward(torch.Tensor(mnist_test.images[786]).view(1, -1))
stop = timeit.default_timer()
print("predicted number: ", np.argmax(prediction.detach().numpy()))
print("correct number ", mnist_test.labels[786])
print(f"run time[s]: {stop-start}")


