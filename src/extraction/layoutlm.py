import os
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Fixed label schema to match SROIE dataset labels ("COMPANY" instead of "VENDOR")
LABELS = [
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

ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


class ReceiptLayoutParser:
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        # Check if model_name is a local directory or online HF model
        is_local = os.path.exists(model_name)

        if is_local:
            self.processor = LayoutLMv3Processor.from_pretrained(
                model_name,
                apply_ocr=False,
                local_files_only=True,
                use_fast=True,
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name, local_files_only=True
            )
        else:
            self.processor = LayoutLMv3Processor.from_pretrained(
                model_name, apply_ocr=False
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
            )

        self.model.eval()

    def prepare_inputs(self, image_path, ocr_results):
        image = Image.open(image_path).convert("RGB")
        words = [item["text"] for item in ocr_results]

        boxes = []
        for item in ocr_results:
            b = item["box"]
            xs = [pt[0] for pt in b]
            ys = [pt[1] for pt in b]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

        w, h = image.size
        normalized_boxes = [
            [
                int(1000 * (box[0] / w)),
                int(1000 * (box[1] / h)),
                int(1000 * (box[2] / w)),
                int(1000 * (box[3] / h)),
            ]
            for box in boxes
        ]

        encoding = self.processor(
            image,
            text=words,
            boxes=normalized_boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        return encoding, words, normalized_boxes

    def predict(self, encoding):
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0)
        return [ID2LABEL[int(p.item())] for p in predictions]
