import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)


class CustomTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def measure_accuracy(model, dataset, device):
    model.eval()
    all_predictions = []
    all_labels = []

    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["positive", "negative"],
        output_dict=True,
    )
    print(f"Model Accuracy: {accuracy:.4f}")
    print(report)
    return all_predictions, all_labels, accuracy, report


def setup_model(model_name, lora_config, load=True, device="cuda", weights=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if load:
        model = PeftModel.from_pretrained(model, weights)
    else:
        model = get_peft_model(model, lora_config)
    model.config.label2id = {0: 0, 1: 1}
    model.config.id2label = {0: "positive", 1: "negative"}
    model.config.label_names = ["labels"]
    model.to(device)
    print(model)
    return model, tokenizer
