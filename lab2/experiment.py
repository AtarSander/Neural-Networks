from sklearn.metrics import (
    f1_score,
    classification_report,
    recall_score,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from torch.utils.data import TensorDataset
from model import Model, train, predict
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


def tune_architecture(
    train_x, train_y, val_x, val_y, architectures, training_setup, device
):
    losses = []
    train_accuracy = []
    val_accuracy = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []
    train_confusion_matrix = []
    val_confusion_matrix = []
    train_roc_curve = []
    val_roc_curve = []

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    )

    for i, architecture in enumerate(architectures):
        model = Model(architecture, device)
        print(f"Training model {i}")
        loss = train(
            model,
            train_dataset,
            training_setup["epochs"],
            training_setup["lr"],
            training_setup["batch_size"],
            training_setup["class_weights"],
        )
        print("Training done")
        losses.append(loss)
        train_y_preds = predict(model, train_x)
        train_accuracy.append(accuracy_score(train_y_preds, train_y))
        train_precision.append(precision_score(train_y, train_y_preds, average="micro"))
        train_recall.append(recall_score(train_y, train_y_preds, average="micro"))
        train_f1.append(f1_score(train_y, train_y_preds, average="micro"))
        train_confusion_matrix.append(confusion_matrix(train_y, train_y_preds))

        val_y_preds = predict(model, val_x)
        val_accuracy.append(accuracy_score(val_y_preds, val_y))
        val_precision.append(precision_score(val_y, val_y_preds, average="micro"))
        val_recall.append(recall_score(val_y, val_y_preds, average="micro"))
        val_f1.append(f1_score(val_y, val_y_preds, average="micro"))
        val_confusion_matrix.append(confusion_matrix(val_y, val_y_preds))

        # ROC curves for each class against all others
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(train_x).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(train_y == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data.append((fpr, tpr, roc_auc))
            train_roc_curve.append(roc_data)

        with torch.no_grad():
            logits = model(torch.tensor(val_x).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(val_y == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data.append((fpr, tpr, roc_auc))
            val_roc_curve.append(roc_data)

    return (
        losses,
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        train_confusion_matrix,
        train_roc_curve,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        val_confusion_matrix,
        val_roc_curve,
    )


def tune_hyperparameters(
    train_x, train_y, val_x, val_y, architecture, hyperparameters, device
):
    losses = []
    train_accuracy = []
    val_accuracy = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []
    train_confusion_matrix = []
    val_confusion_matrix = []
    train_roc_curve = []
    val_roc_curve = []

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    )
    best_model = None
    best_model_f1 = 0

    for i, hyperparameter in enumerate(hyperparameters):
        model = Model(architecture, device)
        print(f"Training model {i}")
        loss = train(
            model,
            train_dataset,
            hyperparameter["epochs"],
            hyperparameter["lr"],
            hyperparameter["batch_size"],
            hyperparameter["class_weights"],
        )
        print("Training done")
        losses.append(loss)
        train_y_preds = predict(model, train_x)
        train_accuracy.append(accuracy_score(train_y_preds, train_y))
        train_precision.append(precision_score(train_y, train_y_preds, average="micro"))
        train_recall.append(recall_score(train_y, train_y_preds, average="micro"))
        train_f1.append(f1_score(train_y, train_y_preds, average="micro"))
        train_confusion_matrix.append(confusion_matrix(train_y, train_y_preds))

        val_y_preds = predict(model, val_x)
        val_accuracy.append(accuracy_score(val_y_preds, val_y))
        val_precision.append(precision_score(val_y, val_y_preds, average="micro"))
        val_recall.append(recall_score(val_y, val_y_preds, average="micro"))
        val_f1.append(f1_score(val_y, val_y_preds, average="micro"))
        val_confusion_matrix.append(confusion_matrix(val_y, val_y_preds))
        if val_f1[-1] > best_model_f1:
            best_model = model
            best_model_f1 = val_f1[-1]
        # ROC curves for each class against all others
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(train_x).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(train_y == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data.append((fpr, tpr, roc_auc))
            train_roc_curve.append(roc_data)

        with torch.no_grad():
            logits = model(torch.tensor(val_x).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(val_y == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data.append((fpr, tpr, roc_auc))
            val_roc_curve.append(roc_data)

    return (
        best_model,
        losses,
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        train_confusion_matrix,
        train_roc_curve,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        val_confusion_matrix,
        val_roc_curve,
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


def plot_confusion_matrices(train_confusion_matrix, val_confusion_matrix):
    fig, axes = plt.subplots(
        nrows=len(train_confusion_matrix),
        ncols=2,
        figsize=(20, len(train_confusion_matrix) * 8),
    )

    for index in range(len(train_confusion_matrix)):
        sns.heatmap(
            train_confusion_matrix[index],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[index, 0],
        )
        axes[index, 0].set_title(f"Train Confusion Matrix for Model {index}")
        axes[index, 0].set_ylabel("Real Label")
        axes[index, 0].set_xlabel("Predicted Label")

        sns.heatmap(
            val_confusion_matrix[index],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[index, 1],
        )
        axes[index, 1].set_title(f"Validation Confusion Matrix for Model {index}")
        axes[index, 1].set_ylabel("Real Label")
        axes[index, 1].set_xlabel("Predicted Label")


def plot_roc_curves(train_roc_curve, val_roc_curve):
    fig, axes = plt.subplots(
        nrows=len(train_roc_curve), ncols=2, figsize=(20, len(train_roc_curve) * 6)
    )
    for index in range(len(train_roc_curve)):
        axes[index, 0].plot([0, 1], [0, 1], "k--", label="Random Guess")
        axes[index, 1].plot([0, 1], [0, 1], "k--", label="Random Guess")

        for i, (fpr, tpr, roc_auc) in enumerate(train_roc_curve[index]):
            axes[index, 0].plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
            axes[index, 0].set_xlabel("False Positive Rate")
            axes[index, 0].set_ylabel("True Positive Rate")
            axes[index, 0].set_title(f"Training ROC Curve for Model {index}")
            axes[index, 0].legend()

        for i, (fpr, tpr, roc_auc) in enumerate(val_roc_curve[index]):
            axes[index, 1].plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
            axes[index, 1].set_xlabel("False Positive Rate")
            axes[index, 1].set_ylabel("True Positive Rate")
            axes[index, 1].set_title(f"Validation ROC Curve for Model {index}")
            axes[index, 1].legend()


def plot_losses(losses):
    for index in range(len(losses)):
        plt.plot(losses[index][0], losses[index][1], label="loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss vs Epoch for Model {index}")
        plt.legend()
        plt.show()
