import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset


class IceCreamData:
    def __init__(self):
        self.data = pd.read_csv("IceCreamData.csv")

    def get_temperature(self):
        return self.data.loc[:, "Temperature"].to_numpy()

    def get_revenue(self):
        return self.data.loc[:, "Revenue"].to_numpy()


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
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.01)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)

    def forward(self, inputs):
        return self.linear(inputs)

    def backward(self, train_loader, epoch, num_epochs):
        self.train()
        cumulative_loss = 0

        for x_values, y_values in train_loader:
            prediction = self.linear(x_values)
            loss = self.loss_function(prediction, y_values)
            loss.backward()
            self.optimizer_function.step()
            self.optimizer_function.zero_grad()
            cumulative_loss += loss.item()

        print(f"Epoch [{epoch + 1:03}/{num_epochs:3}] | Train Loss: {cumulative_loss / len(train_loader):.4f}")


model = LinearRegressionModel()
print("1. random weights from initialization: ", model.state_dict())


iceCreamData = IceCreamData()
dataset = RegressionDataset(iceCreamData.get_temperature(), iceCreamData.get_revenue())

train_dataset, test_dataset = random_split(dataset, lengths=[0.75, 0.25])
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

print()
print("------------- Training the model -------------")
model = LinearRegressionModel()
num_epochs = 10
for epoch in range(num_epochs):
    model.backward(train_loader, epoch, num_epochs)



print()
print("------------- Saving and loading the weights -------------")
print("2. trained weights: ", model.state_dict())
torch.save(model.state_dict(), "mynet-weights.pt") # save weights
model_loaded = LinearRegressionModel()
state = torch.load("mynet-weights.pt")
model_loaded.load_state_dict(state)
print("3. loaded trained weights : ", model_loaded.state_dict()) # loaded weights
model_loaded.eval()

x_calculate = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
y_prediction = []
for temperatur in x_calculate:
    prediction = model_loaded.linear(torch.Tensor([temperatur]))
    y_prediction.append(prediction.tolist()[0])
print("4. Predictions from loaded trained weights: ", y_prediction)

print()
print("------------- Saving and loading the complete model using scripting -------------")
scripted = torch.jit.script(model)
scripted.save("mynet-complete-models.pt") # save model

model_loaded_from_script = torch.jit.load("mynet-complete-models.pt")
print("5. Weights of loaded model from script : ", model_loaded_from_script.state_dict()) #
model_loaded_from_script.eval()

x_calculate = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
y_prediction = []
for temperatur in x_calculate:
    prediction = model_loaded_from_script.linear(torch.Tensor([temperatur]))
    y_prediction.append(prediction.tolist()[0])
print("6. Predictions from loaded model from script: ", y_prediction)

print()
print("------------- Saving and loading the complete model using tracing -------------")
example = torch.rand(1)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("mynet-complete-model-traced.pt") # save model

model_loaded_from_tracing = torch.jit.load("mynet-complete-model-traced.pt")
print("7. Weights of loaded model from tracing : ", model_loaded_from_tracing.state_dict()) #
model_loaded_from_tracing.eval()

x_calculate = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
y_prediction = []
for temperatur in x_calculate:
    prediction = model_loaded_from_script.linear(torch.Tensor([temperatur]))
    y_prediction.append(prediction.tolist()[0])
print("8. Predictions from loaded model from tracing: ", y_prediction)