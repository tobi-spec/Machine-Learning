import torch
import torch.nn as nn
from yaml_parser import get_hyperparameters

hyperparameters: dict = get_hyperparameters("airline_passengers_pytorch_ff_hyperparameter.yaml")


class FeedForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(30, 50)
        self.linear2 = nn.Linear(50, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=hyperparameters["learning_rate"])

    def forward(self, inputs):
        inputs = self.linear1(inputs)
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

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {cumulative_loss / len(train_loader):.4f}")

    def validate(self, val_loader):
        self.eval()
        loss = 0

        with torch.no_grad():
            for x_values, y_values in val_loader:
                prediction = self.forward(x_values)
                loss += self.loss_function(prediction, y_values).item()

        print(f'Validation Loss: {loss / len(val_loader):.4f}')
