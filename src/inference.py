import os
import sys

import torch
from PIL import Image
from paddleocr import PaddleOCR
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Tokenizer,
)

MODEL_PATH = "models/layoutlmv3_receipt_500"
MODEL_NAME = "microsoft/layoutlmv3-base"

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


OCR_ENGINE = PaddleOCR(
    use_textline_orientation=False,
    lang="en",
    device="cpu",
    enable_mkldnn=False,
)


def load_model():
    """Load the trained LayoutLMv3 model and its tokenizer."""
    tokenizer = LayoutLMv3Tokenizer.from_pretrained(
        MODEL_NAME,
        add_prefix_space=True,
    )

    image_processor = LayoutLMv3ImageProcessor.from_pretrained(MODEL_NAME)
    image_processor.apply_ocr = False

    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    return model, tokenizer, image_processor


def normalize_box(box, width, height):
    """Convert pixel coordinates to LayoutLMv3's 0-1000 coordinate system."""
    x1, y1, x2, y2 = box

    return [
        max(0, min(1000, int(x1 / width * 1000))),
        max(0, min(1000, int(y1 / height * 1000))),
        max(0, min(1000, int(x2 / width * 1000))),
        max(0, min(1000, int(y2 / height * 1000))),
    ]


def run_ocr(image_path):
    """Run PaddleOCR and return OCR words with their bounding boxes."""
    result = OCR_ENGINE.predict(image_path)

    words = []
    boxes = []

    for res in result:
        rec_texts = res.get("rec_texts", [])
        rec_boxes = res.get("rec_boxes", [])

        for text, box in zip(rec_texts, rec_boxes):
            text = str(text).strip()

            if not text:
                continue

            words.append(text)
            boxes.append(box.tolist() if hasattr(box, "tolist") else box)

    return words, boxes


def predict_words(image_path):
    """Predict one LayoutLMv3 label for each original OCR word."""
    model, tokenizer, image_processor = load_model()

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    words, pixel_boxes = run_ocr(image_path)

    normalized_boxes = [normalize_box(box, width, height) for box in pixel_boxes]

    # Keep this consistent with the training dataset.
    max_words = 250
    words = words[:max_words]
    normalized_boxes = normalized_boxes[:max_words]

    token_ids = []
    token_boxes = []
    word_indices = []

    for word_index, (word, box) in enumerate(zip(words, normalized_boxes)):
        encoded = tokenizer(
            [word],
            boxes=[box],
            add_special_tokens=False,
        )

        subword_ids = encoded["input_ids"]

        # Handle tokenizers that return [[...]] instead of [...].
        if subword_ids and isinstance(subword_ids[0], list):
            subword_ids = subword_ids[0]

        if not subword_ids:
            subword_ids = [tokenizer.unk_token_id or 0]

        for subword_id in subword_ids:
            token_ids.append(subword_id)
            token_boxes.append(box)
            word_indices.append(word_index)

    # Match the maximum sequence length used during training.
    max_tokens = 512 - 2

    token_ids = token_ids[:max_tokens]
    token_boxes = token_boxes[:max_tokens]
    word_indices = word_indices[:max_tokens]

    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    input_ids = [bos_token_id] + token_ids + [eos_token_id]
    bbox = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    attention_mask = [1] * len(input_ids)

    image_encoding = image_processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        "input_ids": torch.tensor(
            [input_ids],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [attention_mask],
            dtype=torch.long,
        ),
        "bbox": torch.tensor(
            [bbox],
            dtype=torch.long,
        ),
        "pixel_values": image_encoding["pixel_values"],
    }

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)[0].tolist()

    # Ignore BOS because the first prediction belongs to the first token.
    token_predictions = predictions[1 : len(token_ids) + 1]

    # Keep only the prediction from the first subword of each word.
    word_predictions = {}

    for word_index, prediction in zip(
        word_indices,
        token_predictions,
    ):
        if word_index not in word_predictions:
            word_predictions[word_index] = prediction

    print("\nWORD-LEVEL PREDICTIONS")
    print("=" * 60)

    for index, word in enumerate(words):
        label_id = word_predictions.get(index, 0)
        label = LABEL_LIST[label_id]

        print(f"{word:<35} {label}")

    fields = extract_fields(words, word_predictions)

    print("\nEXTRACTED FIELDS")
    print("=" * 60)
    print(f"Company: {fields['company']}")
    print(f"Date:    {fields['date']}")
    print(f"Address: {fields['address']}")
    print(f"Total:   {fields['total']}")

    return fields


def extract_fields(words, word_predictions):
    """Convert word-level predictions into receipt fields."""
    company_candidates = []
    address_candidates = []
    date_candidates = []
    total_candidates = []

    current_field = None
    current_words = []

    def save_current():
        if not current_words or current_field is None:
            return

        value = " ".join(current_words)

        if current_field == "company":
            company_candidates.append(value)
        elif current_field == "address":
            address_candidates.append(value)
        elif current_field == "date":
            date_candidates.append(value)
        elif current_field == "total":
            total_candidates.append(value)

    for index, word in enumerate(words):
        label_id = word_predictions.get(index, 0)
        label = LABEL_LIST[label_id]

        if label == "B-COMPANY":
            save_current()
            current_field = "company"
            current_words = [word]

        elif label == "I-COMPANY" and current_field == "company":
            current_words.append(word)

        elif label == "B-DATE":
            save_current()
            current_field = "date"
            current_words = [word]

        elif label == "I-DATE" and current_field == "date":
            current_words.append(word)

        elif label == "B-ADDRESS":
            save_current()
            current_field = "address"
            current_words = [word]

        elif label == "I-ADDRESS" and current_field == "address":
            current_words.append(word)

        elif label == "B-TOTAL":
            save_current()
            current_field = "total"
            current_words = [word]

        else:
            save_current()
            current_field = None
            current_words = []

    save_current()

    # Prefer the longest company candidate.
    company = max(company_candidates, key=len, default="")

    address = max(address_candidates, key=len, default="")
    date = max(date_candidates, key=len, default="")

    total = total_candidates[0] if total_candidates else ""

    return {
        "company": company,
        "date": date,
        "address": address,
        "total": total,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m src.inference <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    predict_words(image_path)


if __name__ == "__main__":
    main()
