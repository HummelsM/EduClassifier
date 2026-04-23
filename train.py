import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

# Load dataset
df = pd.read_csv(
    r'D://Yumna Folder Extra//My Own Data//data//my_own_data.csv',
    quotechar='"',      # treats everything inside quotes as one column
    on_bad_lines='skip' # optional, in case some rows are still broken
)

# Encode labels
print(df['label'].value_counts())
labels = df["label"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label_id"] = df["label"].map(label2id)

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_id"])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["query"], padding="max_length", truncation=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_encodings = tokenizer(
    train_df["query"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_df["query"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

    
train_dataset = Dataset(train_encodings, train_df["label_id"].tolist())
val_dataset = Dataset(val_encodings, val_df["label_id"].tolist())

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,              # lower = more stable
    per_device_train_batch_size=2,   # small dataset
    per_device_eval_batch_size=2,
    num_train_epochs=2,              # not too high
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    dataloader_pin_memory=False
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    report = classification_report(labels, preds, output_dict=True)
    print(accuracy=report["accuracy"])
    print(f1_score=report["weighted avg"]["f1-score"])
    return {
        "accuracy": report["accuracy"],
        "f1": report["weighted avg"]["f1-score"]
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained("models/")
tokenizer.save_pretrained("models/")

# Save label mapping
import json
with open("models/label_map.json", "w") as f:
    json.dump(id2label, f)

print("Training complete. Model saved.")
