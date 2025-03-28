import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            if "dropout" in layer:
                self.layers.append(nn.Dropout(layer["dropout"]))
            self.layers.append(nn.Linear(*layer["linear"]))
            if "batch_norm" in layer:
                self.layers.append(nn.BatchNorm1d(layer["batch_norm"]))
            if "activation" in layer:
                self.layers.append(nn.ReLU())
        self.initialize_weights(layers)
        self.to(device)
        self.device = device

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

    def initialize_weights(self, layer_dicts):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i == len(self.layers) - 1:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)


def train(
    model, training_dataset, epochs, lr, batch_size, class_weights=None, verbose=False
):
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
    else:
        criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    model.train()
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
    return (list(range(epochs)), losses)


def predict(model, train_x):
    model.eval()
    train_x = torch.from_numpy(train_x).float().to(model.device)
    logits = model(train_x).cpu().detach().numpy()
    y_preds = np.argmax(logits, axis=1)
    return y_preds
