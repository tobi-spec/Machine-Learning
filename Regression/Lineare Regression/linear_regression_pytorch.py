import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.datasets import make_regression

# Source https://datagy.io/pytorch-linear-regression/

class RegressionDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.from_numpy(x.astype("float32"))
        self.y = torch.from_numpy(y.astype("float32"))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index].unsqueeze(0)


class LinearRegressionModel(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.linear = nn.Linear(
                    in_features=in_features,
                    out_features=out_features
                )

    def forward(self, inputs):
        return self.linear(inputs)


def train(model, train_loader, loss_function, optimizer, epoch, num_epochs):
    model.train()
    train_loss = 0.0

    for x_values, y_values in train_loader:
        prediction = model(x_values)
        loss = loss_function(prediction, y_values)
        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = train_loss/len(train_loader)
    print(f"Epoch [{epoch+1:03}/{num_epochs:3}] | Train Loss: {average_loss:.4f}")
    train_losses.append(train_loss/len(train_loader))


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    val_losses.append(avg_loss)



bias = 10
X_numpy, y_numpy, coef = make_regression(
    n_samples=5000,
    n_features=1,
    n_targets=1,
    noise=5,
    bias=bias,
    coef=True,
    random_state=42
)
dataset = RegressionDataset(X_numpy, y_numpy)

train_dataset, test_dataset = random_split(dataset, lengths=[0.8, 0.2])


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=True
)

model = LinearRegressionModel()
loss_function = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

train_losses = []
val_losses = []

num_epochs = 30
for epoch in range(num_epochs):
    train(model, train_loader, loss_function, optimiser, epoch, num_epochs)
    validate(model, test_loader, loss_function)

