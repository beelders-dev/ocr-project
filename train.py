import json
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import ReceiptDataset
from src.extraction.layoutlm import ReceiptLayoutParser


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Initialize Parser & Model
    parser = ReceiptLayoutParser(model_name="microsoft/layoutlmv3-base")
    model = parser.model.to(device)
    model.train()

    # 2. Load Pre-processed JSON Dataset
    dataset_json = "data/processed_dataset.json"
    if not os.path.exists(dataset_json):
        print(f"Error: {dataset_json} not found. Run dataset_generator.py first.")
        return

    with open(dataset_json, "r", encoding="utf-8") as f:
        dataset_samples = json.load(f)

    if not dataset_samples:
        print("No valid dataset samples found in JSON. Exiting training.")
        return

    # 3. Prepare DataLoader & Optimizer
    dataset = ReceiptDataset(dataset_samples, parser.processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # 4. Training Loop
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        valid_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            loss = outputs.loss

            if torch.isnan(loss):
                print("Warning: Encountered NaN loss. Skipping batch gradient step.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # 5. Save Model Checkpoint
    output_dir = "models/layoutlmv3-receipts"
    model.save_pretrained(output_dir)
    parser.processor.save_pretrained(output_dir)
    print(f"\nModel weights updated successfully in {output_dir}")


if __name__ == "__main__":
    train()
