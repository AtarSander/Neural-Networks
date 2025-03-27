from sklearn.metrics import (
    f1_score,
    classification_report,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
)
from torch.utils.data import TensorDataset
from model import Model, train, predict
import torch


def tune_architecture(
    train_x, train_y, val_x, val_y, architectures, training_setup, device
):
    train_accuracy = []
    val_accuracy = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    )

    for i, architecture in enumerate(architectures):
        model = Model(architecture, device)
        print(f"Training model {i}")
        train(
            model,
            train_dataset,
            training_setup["epochs"],
            training_setup["lr"],
            training_setup["batch_size"],
            training_setup["class_weights"],
        )
        print("Training done")
        train_y_preds = predict(model, train_x)
        train_accuracy.append(accuracy_score(train_y_preds, train_y))
        train_precision.append(precision_score(train_y, train_y_preds, average="micro"))
        train_recall.append(recall_score(train_y, train_y_preds, average="micro"))
        train_f1.append(f1_score(train_y, train_y_preds, average="micro"))

        val_y_preds = predict(model, val_x)
        val_accuracy.append(accuracy_score(val_y_preds, val_y))
        val_precision.append(precision_score(val_y, val_y_preds, average="micro"))
        val_recall.append(recall_score(val_y, val_y_preds, average="micro"))
        val_f1.append(f1_score(val_y, val_y_preds, average="micro"))

    return (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
    )


def tune_hyperparameters(
    train_x, train_y, val_x, val_y, architecture, hyperparameters, device
):
    train_accuracy = []
    val_accuracy = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    )
    for i, hyperparameter in enumerate(hyperparameters):
        model = Model(architecture, device)
        print(f"Training model {i}")
        train(
            model,
            train_dataset,
            hyperparameter["epochs"],
            hyperparameter["lr"],
            hyperparameter["batch_size"],
            hyperparameter["class_weights"],
        )
        print("Training done")
        train_y_preds = predict(model, train_x)
        train_accuracy.append(accuracy_score(train_y_preds, train_y))
        train_precision.append(precision_score(train_y, train_y_preds, average="micro"))
        train_recall.append(recall_score(train_y, train_y_preds, average="micro"))
        train_f1.append(f1_score(train_y, train_y_preds, average="micro"))

        val_y_preds = predict(model, val_x)
        val_accuracy.append(accuracy_score(val_y_preds, val_y))
        val_precision.append(precision_score(val_y, val_y_preds, average="micro"))
        val_recall.append(recall_score(val_y, val_y_preds, average="micro"))
        val_f1.append(f1_score(val_y, val_y_preds, average="micro"))

    return (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
    )


def train_final_model(train_x, train_y, architecture, hyperparameters, device):
    model = Model(architecture, device)
    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    )
    print(f"Training final model")
    train(
        model,
        train_dataset,
        hyperparameters["epochs"],
        hyperparameters["lr"],
        hyperparameters["batch_size"],
        hyperparameters["class_weights"],
    )
    print(f"Training done")
    return model
