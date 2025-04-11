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
from model import Model, train, predict
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


def tune_architecture(
    train_dataset,
    val_dataset,
    architectures,
    training_setup,
    device,
    verbose=False,
    save_weights=False,
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

    train_dataloader = DataLoader(
        train_dataset, batch_size=training_setup["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=training_setup["batch_size"], shuffle=False
    )

    for i, architecture in enumerate(architectures):
        model = Model(architecture, device)
        print(f"Training model {i}")
        loss = train(
            model,
            train_dataloader,
            training_setup["epochs"],
            training_setup["lr"],
            training_setup["batch_size"],
            training_setup["class_weights"],
            verbose=verbose,
            save_checkpoints=save_weights,
            identifier=f"t_ar_{i}",
        )
        print("Training done")
        model.eval()
        losses.append(loss)
        train_y_preds, train_labels = predict(model, train_dataloader)
        train_accuracy.append(accuracy_score(train_labels, train_y_preds))
        train_precision.append(
            precision_score(train_labels, train_y_preds, average="macro")
        )
        train_recall.append(recall_score(train_labels, train_y_preds, average="macro"))
        train_f1.append(f1_score(train_labels, train_y_preds, average="macro"))
        train_confusion_matrix.append(confusion_matrix(train_labels, train_y_preds))

        val_y_preds, val_labels = predict(model, val_dataloader)
        val_accuracy.append(accuracy_score(val_labels, val_y_preds))
        val_precision.append(precision_score(val_labels, val_y_preds, average="macro"))
        val_recall.append(recall_score(val_labels, val_y_preds, average="macro"))
        val_f1.append(f1_score(val_labels, val_y_preds, average="macro"))
        val_confusion_matrix.append(confusion_matrix(val_labels, val_y_preds))
        print(classification_report(val_labels, val_y_preds))

        with torch.no_grad():
            logits = train_y_preds
            print(logits, logits.shape)
            probs = (
                F.softmax(torch.from_numpy(logits), dim=1).cpu().numpy()
            )  # i know np.darray->tensor->np.darray is ugly, but dont want to use scipy just for softmax
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(train_labels == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data.append((fpr, tpr, roc_auc))
            train_roc_curve.append(roc_data)

        with torch.no_grad():
            logits = val_y_preds
            probs = F.softmax(torch.from_numpy(logits), dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(val_labels == i, probs[:, i])
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
    train_dataset, val_dataset, architecture, hyperparameters, device, verbose=False
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

    train_labels = [label for _, label in train_dataset]
    train_images = [image for image, _ in train_dataset]

    val_labels = [label for _, label in val_dataset]
    val_images = [image for image, _ in val_dataset]

    best_model = None
    best_model_recall = 0

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
            verbose=verbose,
        )
        print("Training done")
        losses.append(loss)
        train_y_preds = predict(model, train_images)
        train_accuracy.append(accuracy_score(train_labels, train_y_preds))
        train_precision.append(
            precision_score(train_labels, train_y_preds, average="macro")
        )
        train_recall.append(recall_score(train_labels, train_y_preds, average="macro"))
        train_f1.append(f1_score(train_labels, train_y_preds, average="macro"))
        train_confusion_matrix.append(confusion_matrix(train_labels, train_y_preds))

        val_y_preds = predict(model, val_images)
        val_accuracy.append(accuracy_score(val_labels, val_y_preds))
        val_precision.append(precision_score(val_labels, val_y_preds, average="macro"))
        val_recall.append(recall_score(val_labels, val_y_preds, average="macro"))
        val_f1.append(f1_score(val_labels, val_y_preds, average="macro"))
        val_confusion_matrix.append(confusion_matrix(val_labels, val_y_preds))
        print(classification_report(val_labels, val_y_preds))

        if val_recall[-1] > best_model_recall:
            best_model = model
            best_model_recall = val_recall[-1]

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(train_images).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(train_labels == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data.append((fpr, tpr, roc_auc))
            train_roc_curve.append(roc_data)

        with torch.no_grad():
            logits = model(torch.tensor(val_images).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            n_classes = probs.shape[1]
            roc_data = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(val_labels == i, probs[:, i])
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
