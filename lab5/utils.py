from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch


def histogram(data):
    hist = {}
    for item in data:
        hist[item[1]] = hist.get(item[1], 0) + 1
    return hist


def len_histogram(data):
    hist = {}
    for item in data:
        length = len(item[0])
        hist[length] = hist.get(length, 0) + 1
    return hist


def collate_fn(batch, pad_value=-2):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])

    sequences = [torch.tensor(seq).unsqueeze(-1).float() for seq in sequences]

    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    labels = torch.tensor(labels)

    return padded_seqs, lengths, labels


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        return torch.tensor(sequence), torch.tensor(label)


def predict_final(model, test_loader, device="cpu"):
    model.eval().to(device)
    predictions = []
    with torch.no_grad():
        for x, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            outputs = model(x, lengths)
            _, preds = torch.max(outputs, 1)
            predictions.extend([int(p) for p in preds.detach().to("cpu").numpy()])
    return predictions


def collate_fn_test(sequences, pad_value=-2):
    lengths = torch.tensor([len(seq) for seq in sequences])

    sequences = [torch.tensor(seq).unsqueeze(-1).float() for seq in sequences]

    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=pad_value)

    return padded_seqs, lengths


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return torch.tensor(sequence)
