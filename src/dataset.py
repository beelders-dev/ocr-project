import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor

LABEL_LIST = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


class ReceiptDataset(Dataset):
    def __init__(self, dataset, processor=None, max_length=512):
        self.dataset = dataset
        self.max_length = max_length

        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            "microsoft/layoutlmv3-base", add_prefix_space=True
        )
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        self.image_processor.apply_ocr = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # 1. Handle image extraction robustly
        image = sample.get("image")

        if image is None:
            image_path = sample.get("image_path", "")

            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                raise FileNotFoundError(
                    f"Image not found for sample {idx}: " f"{image_path}"
                )

        # 2. Extract Hugging Face dataset keys correctly
        tokens = sample.get("tokens", sample.get("words", []))
        bboxes = sample.get("bboxes", sample.get("boxes", sample.get("bbox", [])))
        ner_tags = sample.get("ner_tags", sample.get("labels", []))

        if not tokens or not bboxes:
            tokens = ["N/A"]
            bboxes = [[0, 0, 10, 10]]
            ner_tags = [0]

        min_len = min(len(tokens), len(bboxes), len(ner_tags))
        tokens = [str(t) for t in tokens[:min_len]]
        bboxes = bboxes[:min_len]
        ner_tags = ner_tags[:min_len]

        word_labels = []
        for lbl in ner_tags:
            if isinstance(lbl, str):
                word_labels.append(LABEL2ID.get(lbl, 0))
            else:
                word_labels.append(int(lbl))

        normalized_boxes = []

        for box in bboxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                normalized_boxes.append([0, 0, 1, 1])
                continue

            x1, y1, x2, y2 = [max(0, min(1000, int(value))) for value in box]

            if x2 <= x1:
                x2 = min(1000, x1 + 1)

            if y2 <= y1:
                y2 = min(1000, y1 + 1)

            normalized_boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

        max_words = 250
        tokens = tokens[:max_words]
        normalized_boxes = normalized_boxes[:max_words]
        word_labels = word_labels[:max_words]

        # 3. Manual token and label alignment compatible with the slow tokenizer
        token_ids_list = []
        token_boxes_list = []
        token_labels_list = []

        for word, box, label in zip(tokens, normalized_boxes, word_labels):
            subword_ids = self.tokenizer(
                [word],
                boxes=[box],
                add_special_tokens=False,
            )["input_ids"]
            if not subword_ids:
                subword_ids = [self.tokenizer.unk_token_id or 0]

            for i, sub_id in enumerate(subword_ids):
                token_ids_list.append(sub_id)
                token_boxes_list.append(box)
                token_labels_list.append(label if i == 0 else -100)

        max_tokens = self.max_length - 2
        token_ids_list = token_ids_list[:max_tokens]
        token_boxes_list = token_boxes_list[:max_tokens]
        token_labels_list = token_labels_list[:max_tokens]

        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else 0
        )
        eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else 2
        )

        final_input_ids = [bos_token_id] + token_ids_list + [eos_token_id]
        final_boxes = [[0, 0, 0, 0]] + token_boxes_list + [[1000, 1000, 1000, 1000]]
        final_labels = [-100] + token_labels_list + [-100]
        attention_mask = [1] * len(final_input_ids)

        pad_len = self.max_length - len(final_input_ids)
        if pad_len > 0:
            final_input_ids += [self.tokenizer.pad_token_id or 1] * pad_len
            final_boxes += [[0, 0, 0, 0]] * pad_len
            final_labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        # 4. Process visual features
        image_encoding = self.image_processor(images=image, return_tensors="pt")

        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bbox": torch.tensor(final_boxes, dtype=torch.long),
            "pixel_values": image_encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(final_labels, dtype=torch.long),
        }
