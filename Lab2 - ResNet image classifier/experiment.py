from sklearn.metrics import (
    f1_score,
    classification_report,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from model import Model, train, predict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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


def resume_tuning(
    model_path,
    train_dataset,
    val_dataset,
    architecture,
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

    model = Model(architecture, device)
    model.load_weights(model_path)
    print(f"Training model")
    loss = train(
        model,
        train_dataloader,
        training_setup["epochs"],
        training_setup["lr"],
        training_setup["batch_size"],
        training_setup["class_weights"],
        verbose=verbose,
        save_checkpoints=save_weights,
        identifier=f"t_ar_resume",
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


def plot_losses(losses):
    for index in range(len(losses)):
        plt.plot(losses[index][0], losses[index][1], label="loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss vs Epoch")
        plt.legend()
        plt.show()
