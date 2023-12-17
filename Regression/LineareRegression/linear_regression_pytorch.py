import torch
import timeit
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


start = timeit.default_timer()


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
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.5)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)

    def forward(self, inputs):
        return self.linear(inputs)

    def backward(self, train_loader, epoch, num_epochs):
        self.train()

        for x_values, y_values in train_loader:
            prediction = self.linear(x_values)
            loss = self.loss_function(prediction, y_values)
            loss.backward()
            self.optimizer_function.step()
            self.optimizer_function.zero_grad()

        print(f"Epoch [{epoch + 1:03}/{num_epochs:3}] | Train Loss: {loss.item():.4f}")

    def validate(self, val_loader):
        self.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.linear(inputs)
                loss = self.loss_function(outputs, targets)

        print(f'Validation Loss: {loss.item():.4f}')


data = pd.read_csv("./IceCreamData.csv", delimiter=",")
x_values = data[["Temperature"]].to_numpy()
y_values = data["Revenue"].to_numpy()

dataset = RegressionDataset(x_values, y_values)
train_dataset, test_dataset = random_split(dataset, lengths=[0.75, 0.25])
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

model = LinearRegressionModel()
num_epochs = 25
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)
    model.validate(test_loader)

x_calculate = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
y_prediction = []
for temperatur in x_calculate:
    prediction = model.linear(torch.Tensor([temperatur]))
    y_prediction.append(prediction.tolist()[0])
stop = timeit.default_timer()

plt.scatter(x_values, y_values, color="grey")
plt.plot(x_calculate, y_prediction, color="red")
plt.xlabel("revenue [dolars]")
plt.ylabel("temperature [degC]")
plt.title('Linear regression with Pytorch')
plt.figtext(0.2, 0.8, f"run time[s]: {stop-start}")
plt.savefig("./img/linear_regression_pytorch")
plt.show()
