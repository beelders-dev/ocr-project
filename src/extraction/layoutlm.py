import os
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

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
        is_local = os.path.exists(model_name)

        if is_local:
            self.processor = LayoutLMv3Processor.from_pretrained(
                model_name,
                apply_ocr=False,
                local_files_only=True,
                use_fast=True,
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        else:
            self.processor = LayoutLMv3Processor.from_pretrained(
                model_name,
                apply_ocr=False,
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )

        self.model.eval()

    def normalize_box(self, box, width, height):
        """Convert an image bounding box to LayoutLM's 0-1000 coordinate system."""
        x1, y1, x2, y2 = box

        normalized = [
            int(1000 * x1 / width),
            int(1000 * y1 / height),
            int(1000 * x2 / width),
            int(1000 * y2 / height),
        ]

        return [max(0, min(1000, value)) for value in normalized]

    def prepare_inputs(self, image_path, ocr_results):
        image = Image.open(image_path).convert("RGB")
        words = [item["text"] for item in ocr_results]

        width, height = image.size
        normalized_boxes = []

        for item in ocr_results:
            points = item["box"]

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]

            box = [
                min(xs),
                min(ys),
                max(xs),
                max(ys),
            ]

            normalized_boxes.append(self.normalize_box(box, width, height))

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

    def predict(self, encoding, words):
        """Predict one LayoutLM label for each original OCR word."""
        with torch.no_grad():
            outputs = self.model(**encoding)
            token_predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            ).squeeze(0)

        # Map LayoutLM tokens back to the original OCR words.
        word_ids = encoding.word_ids(batch_index=0)

        word_predictions = []
        seen_words = set()

        for token_index, word_index in enumerate(word_ids):
            if word_index is None:
                continue

            # Only use the first token belonging to each OCR word.
            if word_index in seen_words:
                continue

            seen_words.add(word_index)

            label_id = int(token_predictions[token_index].item())
            label = ID2LABEL[label_id]

            word_predictions.append(
                {
                    "text": words[word_index],
                    "label": label,
                }
            )

        return word_predictions
