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

    optimizer = AdamW(model.parameters(), lr=1e-5)
    accumulation_steps = 4

    # 4. Training Loop
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        valid_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}

            # --- DEBUG SECTION ---
            if step == 0 and epoch == 0:
                labels_tensor = inputs.get("labels", None)
                print("\n--- DEBUG BATCH INFO ---")
                print("Batch Keys:", list(inputs.keys()))
                if labels_tensor is not None:
                    print("Labels shape:", labels_tensor.shape)
                    print("Unique values in labels:", torch.unique(labels_tensor))
                    print("First 30 label values:", labels_tensor[0][:30].tolist())
                else:
                    print("CRITICAL: 'labels' key is missing from inputs!")
                print("------------------------\n")
            # ---------------------

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            valid_batches += 1

            if step % 20 == 0 and valid_batches > 0:
                print(
                    f"Epoch {epoch + 1} | Step {step}/{len(dataloader)} | Loss: {loss.item() * accumulation_steps:.4f}"
                )

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        print(
            f"--- Epoch {epoch + 1}/{epochs} Complete - Avg Loss: {avg_loss:.4f} ---\n"
        )

    # 5. Save Model Checkpoint
    output_dir = "models/layoutlmv3-receipts"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    parser.processor.save_pretrained(output_dir)
    print(f"\nModel weights updated successfully in {output_dir}")


if __name__ == "__main__":
    train()
