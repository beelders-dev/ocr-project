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


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Total receipts: {len(dataset)}")

    random.seed(RANDOM_SEED)
    random.shuffle(dataset)

    split_index = int(len(dataset) * TRAIN_RATIO)

    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

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


def run_training_smoke_test(train_loader):
    print("\n" + "=" * 60)
    print("TRAINING SMOKE TEST")
    print("=" * 60)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label={i: label for i, label in enumerate(LABEL_LIST)},
        label2id={label: i for i, label in enumerate(LABEL_LIST)},
    )

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    batch = next(iter(train_loader))

    batch = {key: value.to(device) for key, value in batch.items()}

    print(f"Input IDs shape:    {batch['input_ids'].shape}")
    print(f"Bounding boxes:     {batch['bbox'].shape}")
    print(f"Pixel values:       {batch['pixel_values'].shape}")
    print(f"Labels shape:       {batch['labels'].shape}")

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        bbox=batch["bbox"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"],
    )

    loss = outputs.loss

    print(f"Loss before update: {loss.item():.4f}")

    loss.backward()

    classifier_grad = model.classifier.weight.grad

    if classifier_grad is None:
        raise RuntimeError("No gradient was produced.")

    print(f"Classifier gradient norm: {classifier_grad.norm().item():.6f}")

    optimizer.step()

    print("Optimizer step completed.")
    print("Training smoke test passed.")


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

    run_training_smoke_test(train_loader)


if __name__ == "__main__":
    main()
