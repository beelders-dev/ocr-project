import torch
from torch.utils.data import Dataset
from PIL import Image


class ReceiptDataset(Dataset):
    def __init__(self, samples, processor, max_length=512):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        encoding = self.processor(
            image,
            text=sample["words"],
            boxes=sample["boxes"],
            word_labels=sample["labels"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}
