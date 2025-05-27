from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_input)
        if self.bidirectional:
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            output = self.fc(hidden_cat)
        else:
            output = self.fc(hidden[-1])
        return output


class GRUClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, bidirectional=False
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_input)
        if self.bidirectional:
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            output = self.fc(hidden_cat)
        else:
            output = self.fc(hidden[-1])
        return output


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(
    model,
    dataloader,
    val_dataloader,
    device="cpu",
    epochs=5,
    lr=1e-3,
    patience=5,
    file_path=None,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for x, lengths, y in dataloader:
            x, lengths, y = x.to(device).float(), lengths.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x, lengths)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total_samples += y.size(0)

            total_loss += loss.item()

        epoch_accuracy = correct / total_samples

        predictions, labels = predict(model, val_dataloader, device)
        val_accuracy = evaluate(predictions, labels)
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f} | Train accuracy = {epoch_accuracy* 100:.2f}% | Val accuracy = {val_accuracy * 100:.2f}%"
            )
            if file_path:
                torch.save(model, file_path)
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break


def predict(model, test_loader, device="cpu"):
    model.eval().to(device)
    predictions = []
    labels = []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            outputs = model(x, lengths)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().to("cpu"))
            labels.extend(y.numpy())
    return predictions, labels


def evaluate(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(labels)
    return accuracy


def measure_performance(
    models,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    patience,
    path,
):
    results = []
    for i, model in enumerate(models):
        print("-" * 30, f"Training model {i}", "-" * 30)
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            file_path=path + f"model_{i}.pth",
        )
        train_preds, train_labels = predict(model, train_loader, device)
        train_accuracy = evaluate(train_preds, train_labels)
        val_preds, val_labels = predict(model, val_loader, device)
        val_accuracy = evaluate(val_preds, val_labels)
        results.append((str(model), train_accuracy, val_accuracy))
    return results


def test_models(
    models,
    train_loader,
    val_loader,
    device,
    lrs,
    epochs=100,
    patience=10,
    path="models/",
):
    results = []
    for i, model in enumerate(models):
        print("-" * 30, f"Training model {i}", "-" * 30)
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            lr=lrs[i],
            patience=patience,
            file_path=path + f"final_model_{i}.pth",
        )
        train_preds, train_labels = predict(model, train_loader, device)
        train_accuracy = evaluate(train_preds, train_labels)
        val_preds, val_labels = predict(model, val_loader, device)
        val_accuracy = evaluate(val_preds, val_labels)
        results.append((str(model), train_accuracy, val_accuracy))
    return results
