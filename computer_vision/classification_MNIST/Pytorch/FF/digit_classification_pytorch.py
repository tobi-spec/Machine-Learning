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
        return self.images[index], self.labels[index]


class MNISTClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001)

    def init_weights(self):
        if isinstance(self.model, nn.Linear):
            torch.nn.init.normal_(self.model.weight, mean=0.0, std=1.0)

    def forward(self, inputs):
        return self.model(inputs)

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


mnist_training = MNISTDataset(images="./data/train-images.idx3-ubyte", labels="./data/train-labels.idx1-ubyte")
mnist_test = MNISTDataset(images="./data/t10k-images.idx3-ubyte", labels="./data/t10k-labels.idx1-ubyte")

train_loader = DataLoader(dataset=mnist_training, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=mnist_test, batch_size=32, shuffle=False)

model = MNISTClassificationModel()
model.init_weights()
num_epochs = 15
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)
    model.validate(test_loader)

prediction = model(torch.Tensor(mnist_test.images[786]).view(1, -1))
stop = timeit.default_timer()
print("predicted number: ", np.argmax(prediction.detach().numpy()))
print("correct number ", mnist_test.labels[786])
print(f"run time[s]: {stop - start}")
