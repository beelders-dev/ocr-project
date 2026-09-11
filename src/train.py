import json
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification

from src.dataset import LABEL_LIST, ReceiptDataset

DATASET_PATH = "data/processed_dataset.json"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

BATCH_SIZE = 1
LEARNING_RATE = 5e-5

MODEL_NAME = "microsoft/layoutlmv3-base"

TRAIN_SAMPLES = 100


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Total receipts: {len(dataset)}")

    random.seed(RANDOM_SEED)
    random.shuffle(dataset)

    split_index = int(len(dataset) * TRAIN_RATIO)

    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

    train_data = train_data[:TRAIN_SAMPLES]

    print(f"Training receipts:   {len(train_data)}")
    print(f"Validation receipts: {len(val_data)}")

    return train_data, val_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def print_label_distribution(train_data, val_data):
    def count_labels(data):
        counts = Counter()

        for sample in data:
            for label in sample.get("ner_tags", []):
                if isinstance(label, int):
                    counts[LABEL_LIST[label]] += 1
                else:
                    counts[str(label)] += 1

        return counts

    train_counts = count_labels(train_data)
    val_counts = count_labels(val_data)

    print("\nTraining label distribution:")
    for label in LABEL_LIST:
        print(f"  {label:<15} {train_counts[label]}")

    print("\nValidation label distribution:")
    for label in LABEL_LIST:
        print(f"  {label:<15} {val_counts[label]}")


def inspect_samples(data, num_samples=5):
    print("\n" + "=" * 60)
    print("SAMPLE LABEL INSPECTION")
    print("=" * 60)

    for sample_index, sample in enumerate(data[:num_samples]):
        print(f"\nReceipt {sample_index}")
        print("-" * 60)

        words = sample.get("words", [])
        labels = sample.get("ner_tags", [])

        for word, label in zip(words, labels):
            if isinstance(label, int):
                label_name = LABEL_LIST[label]
            else:
                label_name = str(label)

            if label_name != "O":
                print(f"{word:<30} {label_name}")


def create_dataloaders(train_data, val_data):
    train_dataset = ReceiptDataset(train_data)
    val_dataset = ReceiptDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer):
    model.train()

    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  Step {step}/{len(train_loader)} " f"- Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


def evaluate(model, val_loader):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )

            total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


def create_model():
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label={i: label for i, label in enumerate(LABEL_LIST)},
        label2id={label: i for i, label in enumerate(LABEL_LIST)},
    )

    return model.to(device)


def main():
    train_data, val_data = load_dataset()

    print_label_distribution(train_data, val_data)
    inspect_samples(train_data)

    train_loader, val_loader = create_dataloaders(
        train_data,
        val_data,
    )

    print("\nDataLoaders created successfully.")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model = create_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nStarting training...")

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
    )

    print(f"\nTraining loss: {train_loss:.4f}")

    val_loss = evaluate(
        model,
        val_loader,
    )

    print(f"Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
