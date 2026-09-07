import os
import re
import torch
from PIL import Image, ImageOps
from paddleocr import PaddleOCR
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from src.ocr_engine import OCR_ENGINE

ID2LABEL = {
    0: "O",
    1: "B-VENDOR",
    3: "B-DATE",
    5: "B-TOTAL",
    7: "B-ADDRESS",
}


def clean_extracted_value(label: str, text: str) -> str:
    """Strips prefix noise and isolates exact values using regex."""
    if label == "B-DATE":
        # Extract standard date formats (MM/DD/YYYY, DD-MM-YYYY, YYYY/MM/DD, etc.)
        match = re.search(
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", text
        )
        return match.group(0) if match else text

    elif label == "B-TOTAL":
        match = re.search(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b", text)
        return match.group(0) if match else ""

    return text.strip()


def run_inference(image_path: str, model_path: str = "models/layoutlmv3-receipts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    res = OCR_ENGINE.predict(image_path)
    words, norm_boxes = [], []

    for r in res:
        res_data = r.json["res"] if hasattr(r, "json") and "res" in r.json else r
        boxes = res_data.get("rec_polys", res_data.get("dt_polys", []))
        texts = res_data.get("rec_text", res_data.get("rec_texts", []))

        for b, t in zip(boxes, texts):
            box_list = b.tolist() if hasattr(b, "tolist") else b

            if isinstance(box_list[0], (list, tuple)):
                xs = [pt[0] for pt in box_list]
                ys = [pt[1] for pt in box_list]
            else:
                xs = [box_list[i] for i in range(0, len(box_list), 2)]
                ys = [box_list[i] for i in range(1, len(box_list), 2)]

            nx1 = max(0, min(1000, int(1000 * (min(xs) / w))))
            ny1 = max(0, min(1000, int(1000 * (min(ys) / h))))
            nx2 = max(nx1, min(1000, int(1000 * (max(xs) / w))))
            ny2 = max(ny1, min(1000, int(1000 * (max(ys) / h))))

            words.append(str(t))
            norm_boxes.append([nx1, ny1, nx2, ny2])

    if not words:
        print(f"No text detected in {image_path}")
        return

    # Encode inputs
    encoding = processor(
        img,
        words,
        boxes=norm_boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    ).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # Map subword predictions back to raw OCR tokens
    word_labels = {}
    for pred_id, word_idx in zip(predictions, word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_labels:
            word_labels[word_idx] = pred_id

    # Print extracted and cleaned entities
    print(f"\n--- Extracted Receipt Entities: {image_path} ---")
    extracted_data = {}

    for idx, word in enumerate(words):
        pred_id = word_labels.get(idx, 0)
        label = ID2LABEL.get(pred_id, "O")
        print(f"[{label:<9} (ID {pred_id})] : {word}")

        if label != "O":
            cleaned_val = clean_extracted_value(label, word)

            # Store grouped unique values per entity type
            if label not in extracted_data:
                extracted_data[label] = []
            if cleaned_val not in extracted_data[label]:
                extracted_data[label].append(cleaned_val)

    # Structured key-value printout
    for label, values in extracted_data.items():
        print(f"{label:<12}: {', '.join(values)}")


if __name__ == "__main__":
    sample_image = "data/raw/sample_receipt.jpg"
    if os.path.exists(sample_image):
        run_inference(sample_image)
    else:
        print(f"File not found: {sample_image}")
