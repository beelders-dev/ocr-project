import os
import sys
import time
import ast
import subprocess
import time
import psutil


import torch
from PIL import Image
from paddleocr import PaddleOCR
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Tokenizer,
)
from src.extraction.receipt_fields import extract_receipt_fields
from src.preprocessing.image import resize_image_if_needed

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "models/layoutlmv3_receipt_500",
)
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


MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None


def load_model():
    """Load LayoutLMv3 once and reuse it for subsequent receipt requests."""
    global MODEL, TOKENIZER, IMAGE_PROCESSOR

    if MODEL is None:
        print("[BENCH] Loading LayoutLMv3 model...", flush=True)

        TOKENIZER = LayoutLMv3Tokenizer.from_pretrained(
            MODEL_NAME,
            add_prefix_space=True,
        )

        IMAGE_PROCESSOR = LayoutLMv3ImageProcessor.from_pretrained(MODEL_NAME)
        IMAGE_PROCESSOR.apply_ocr = False

        MODEL = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
        MODEL.eval()

        print("[BENCH] LayoutLMv3 model loaded.", flush=True)

    return MODEL, TOKENIZER, IMAGE_PROCESSOR


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
    """Run PaddleOCR in a separate process and return OCR words with boxes."""
    start = time.perf_counter()

    log_memory("before OCR")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.ocr_worker",
            image_path,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start

    log_memory("after OCR")
    print(f"[BENCH] PaddleOCR: {elapsed:.2f}s", flush=True)

    if result.returncode != 0:
        print("OCR WORKER FAILED")
        print("RETURN CODE:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError("OCR worker failed")

    words = []
    boxes = []

    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue

        text, box_text = line.split("\t", 1)
        words.append(text)
        boxes.append(ast.literal_eval(box_text))

    print(f"[BENCH] OCR words: {len(words)}", flush=True)

    return words, boxes


def predict_words(image_path):
    """Predict one LayoutLMv3 label for each original OCR word."""
    total_start = time.perf_counter()
    log_memory("start")

    resize_start = time.perf_counter()

    processed_image_path = resize_image_if_needed(image_path)

    resize_time = time.perf_counter() - resize_start
    temporary_image = processed_image_path != image_path

    try:
        image = Image.open(processed_image_path).convert("RGB")
        width, height = image.size

        ocr_start = time.perf_counter()

        words, pixel_boxes = run_ocr(processed_image_path)
        log_memory("after OCR pipeline")

        model_start = time.perf_counter()
        model, tokenizer, image_processor = load_model()
        model_load_time = time.perf_counter() - model_start

        ocr_time = time.perf_counter() - ocr_start

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

        bos_token_id = (
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        )

        eos_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        )

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

        inference_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(**inputs)

        inference_time = time.perf_counter() - inference_start

        print(f"[BENCH] LayoutLMv3: {inference_time:.2f}s", flush=True)
        log_memory("after LayoutLMv3")
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

        predictions = []

        for index, word in enumerate(words):
            label_id = word_predictions.get(index, 0)
            label = LABEL_LIST[label_id]

            predictions.append(
                {
                    "text": word,
                    "label": label,
                }
            )

        fields = extract_receipt_fields(predictions)

        print("\nEXTRACTED FIELDS")
        print("=" * 60)
        print(f"Company:       {fields['company']}")
        print(f"Date:          {fields['date']}")
        print(f"TIN:           {fields['tin']}")
        print(f"Invoice No.:   {fields['invoice_number']}")
        print(f"Vatable Sales: {fields['vatable_sales']}")
        print(f"VAT Amount:    {fields['vat_amount']}")
        print(f"Total:         {fields['total']}")
        print(f"VAT Valid:     {'Yes' if fields['vat_valid'] else 'No'}")
        print(f"Total Valid:   {'Yes' if fields['total_valid'] else 'No'}")

        total_time = time.perf_counter() - total_start

        print("\nPERFORMANCE")
        print("=" * 60)
        print(f"Model loading:   {model_load_time:.2f}s")
        print(f"Image resizing:  {resize_time:.2f}s")
        print(f"PaddleOCR:       {ocr_time:.2f}s")
        print(f"LayoutLMv3:      {inference_time:.2f}s")
        print(f"Total:           {total_time:.2f}s")

        return fields

    finally:
        if temporary_image:
            try:
                os.remove(processed_image_path)
            except FileNotFoundError:
                pass


def log_memory(label):
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss

    for child in process.children(recursive=True):
        try:
            memory += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass

    print(f"[BENCH] {label} memory: {memory / 1024**2:.0f} MB", flush=True)


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
