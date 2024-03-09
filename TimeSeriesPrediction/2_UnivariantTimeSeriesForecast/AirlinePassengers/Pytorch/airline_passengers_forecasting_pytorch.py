import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class AirlinePassengersPreparer:
    def __init__(self):
        self.data = pd.read_csv("../../AirlinePassengers.csv", sep=";")
        self.month = self.data.loc[:, "Month"]
        self.passengers = self.data.loc[:, "Passengers"]
        self.passengers_plus_1 = None

    def create_forecast_sequence(self):
        self.passengers_plus_1 = self.passengers.shift(-1)
        self.passengers_plus_1 = self.passengers_plus_1.iloc[:-1]
        self.data["Passengers+1"] = self.passengers_plus_1
        self.passengers = self.passengers.iloc[:-1]

    def get_train_test(self):
        train = self.data[0:107]
        test = self.data[107:142]
        return train, test


class AirlinePassengersDataset(Dataset):
    def __init__(self, passengers, passengers_plus_1):
        self.passengers = torch.from_numpy(passengers.astype("float32"))
        self.passengers_plus_1 = torch.from_numpy(passengers_plus_1.astype("float32"))

    def __len__(self):
        return len(self.passengers)

    def __getitem__(self, index):
        return self.passengers[index], self.passengers_plus_1[index]


class AirlinePassengersModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 8)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(8, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
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


timeseries = AirlinePassengersPreparer()
timeseries.create_forecast_sequence()
train, test = timeseries.get_train_test()

train_dataset = AirlinePassengersDataset(train.loc[:, "Passengers"].to_numpy(), train.loc[:, "Passengers+1"].to_numpy())
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

test_dataset = AirlinePassengersDataset(test.loc[:, "Passengers"].to_numpy(), test.loc[:, "Passengers+1"].to_numpy())
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = AirlinePassengersModel()
num_epochs = 25
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)
    model.validate(test_loader)

predictions = []
for element in test.loc[:, "Passengers"]:
    prediction = model(torch.Tensor([element]))
    predictions.append(prediction.item())
test["Predictions"] = predictions

test.drop(["Passengers+1"], axis=1).to_csv("./AirlinePassengersResultsPytorch.csv")

plt.plot(train["Month"], train["Passengers"], color="green", label="training")
plt.plot(test["Month"], test["Predictions"], color="red", label="prediction")
plt.plot(test["Month"], test["Passengers"], color="blue", label="test")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./airlinePassengers_pytorch.png")
plt.show()
