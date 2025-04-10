import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            if "dropout" in layer:
                self.layers.append(nn.Dropout(layer["dropout"]))
            if "linear" in layer:
                self.layers.append(nn.Linear(*layer["linear"]))
            if "conv" in layer:
                self.layers.append(nn.Conv2d(*layer["conv"]))
            if "batch_norm" in layer and "linear" in layer:
                self.layers.append(nn.BatchNorm1d(layer["batch_norm"]))
            if "batch_norm" in layer and "conv" in layer:
                self.layers.append(nn.BatchNorm2d(layer["batch_norm"]))
            if "activation" in layer:
                self.layers.append(nn.ReLU())
            if "pool" in layer:
                self.layers.append(nn.MaxPool2d(layer["pool"]))
            if "flatten" in layer:
                self.layers.append(nn.Flatten(start_dim=1))
        self.initialize_weights()
        self.to(device)
        self.device = device

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

    def initialize_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i == len(self.layers) - 1:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class ResidualModel(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            if "dropout" in layer:
                self.layers.append(nn.Dropout(layer["dropout"]))
            if "linear" in layer:
                self.layers.append(nn.Linear(*layer["linear"]))
            if "conv" in layer:
                self.layers.append(nn.Conv2d(*layer["conv"]))
            if "batch_norm" in layer and "linear" in layer:
                self.layers.append(nn.BatchNorm1d(layer["batch_norm"]))
            if "batch_norm" in layer and "conv" in layer:
                self.layers.append(nn.BatchNorm2d(layer["batch_norm"]))
            if "activation" in layer:
                self.layers.append(nn.ReLU())
            if "max_pool" in layer:
                self.layers.append(nn.MaxPool2d(*layer["max_pool"]))
            if "avg_pool" in layer:
                self.layers.append(nn.AvgPool2d(*layer["avg_pool"]))
            if "flatten" in layer:
                self.layers.append(nn.Flatten(start_dim=1))
            if "residual" in layer:
                self.layers.append(ResidualBlock(layer["residual"], device))
        self.initialize_weights()
        self.to(device)
        self.device = device

    def read_sequential_block(self, layers):
        block = nn.Sequential()
        for layer in layers:
            if "dropout" in layer:
                block.append(nn.Dropout(layer["dropout"]))
            if "linear" in layer:
                block.append(nn.Linear(*layer["linear"]))
            if "conv" in layer:
                block.append(nn.Conv2d(*layer["conv"]))
            if "batch_norm" in layer and "linear" in layer:
                block.append(nn.BatchNorm1d(layer["batch_norm"]))
            if "batch_norm" in layer and "conv" in layer:
                block.append(nn.BatchNorm2d(layer["batch_norm"]))
            if "activation" in layer:
                block.append(nn.ReLU())
            if "pool" in layer:
                block.append(nn.MaxPool2d(layer["pool"]))
            if "flatten" in layer:
                block.append(nn.Flatten(start_dim=1))
        return block

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

    def initialize_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i == len(self.layers) - 1:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class ResidualBlock(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.block = nn.Sequential()
        self.layers = layers
        for layer in self.layers:
            if "dropout" in layer:
                self.block.append(nn.Dropout(layer["dropout"]))
            if "linear" in layer:
                self.block.append(nn.Linear(*layer["linear"]))
            if "conv" in layer:
                self.block.append(nn.Conv2d(*layer["conv"]))
            if "batch_norm" in layer and "linear" in layer:
                self.block.append(nn.BatchNorm1d(layer["batch_norm"]))
            if "batch_norm" in layer and "conv" in layer:
                self.block.append(nn.BatchNorm2d(layer["batch_norm"]))
            if "activation" in layer:
                self.block.append(nn.ReLU())
            if "pool" in layer:
                self.block.append(nn.MaxPool2d(layer["pool"]))
            if "flatten" in layer:
                self.block.append(nn.Flatten(start_dim=1))
        if layers[0]["conv"][0] != layers[0]["conv"][1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    layers[0]["conv"][0],
                    layers[-1]["conv"][0],
                    kernel_size=1,
                    stride=layers[0]["conv"][3],
                ),
                nn.BatchNorm2d(layers[-1]["conv"][0]),
            )
        else:
            self.shortcut = nn.Identity()
        self.initialize_weights()
        self.to(device)
        self.device = device

    def forward(self, x):
        residual = x
        for module in self.block:
            x = module(x)
        x += self.shortcut(residual)
        return x

    def initialize_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i == len(self.layers) - 1:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)


def train(
    model,
    training_dataset,
    epochs,
    lr,
    batch_size,
    class_weights=None,
    verbose=False,
    save_checkpoints=False,
    identifier="",
):
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
    else:
        criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    if verbose:
        print("Training started")
    for epoch in range(epochs):
        avg_loss = 0
        for input, label in train_dataloader:
            input, label = input.to(model.device), label.to(model.device)
            optimizer.zero_grad()
            y_pred = model(input)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        losses.append(avg_loss / len(train_dataloader))
        if verbose:
            print(f"Epoch: {epoch}, loss: {avg_loss/len(train_dataloader)}")
        if save_checkpoints:
            model.save_weights(f"models/model_{identifier}.pth")
    return (list(range(epochs)), losses)


def predict(model, images):
    model.eval()
    with torch.no_grad():
        logits = model(images).cpu().detach().numpy()
        y_preds = np.argmax(logits, axis=1)
    return np.array(y_preds)
