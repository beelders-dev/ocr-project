import json
import os
import re
from PIL import Image, ImageOps
from src.extraction.layoutlm import LABEL2ID
from src.ocr_engine import OCR_ENGINE  # Correct single import


def auto_label_ocr(ocr_results, label2id):
    words, raw_boxes, labels = [], [], []

    for item in ocr_results:
        text = str(item["text"]).strip()
        if not text:
            continue

        box = item["box"]
        if isinstance(box[0], (list, tuple)):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
        else:
            xs = [box[i] for i in range(0, len(box), 2)]
            ys = [box[i] for i in range(1, len(box), 2)]

        bbox = [min(xs), min(ys), max(xs), max(ys)]

        # --- REFINED HEURISTICS ---
        # 1. Total: Only match price decimals if the line or surrounding context hints at Total/Cash/Amount
        if (
            re.search(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b", text)
            and "TOTAL" in text.upper()
        ):
            label = label2id.get("B-TOTAL", 0)

        # 2. Date formats (e.g., MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY)
        elif re.search(
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", text
        ):
            label = label2id.get("B-DATE", 0)

        # 3. Address components
        elif re.search(
            r"\b(LAS PINAS|SUBD|BRGY|BARANGAY|STREET|ST|AVE|AVENUE|CITY|CAVITE|SILANG)\b",
            text,
            re.IGNORECASE,
        ):
            label = label2id.get("B-ADDRESS", 0)

        # 4. Vendor Name
        elif re.search(
            r"^(ALFAMART|7-ELEVEN|MINISTOP|LAZADA|SHOPEE)", text, re.IGNORECASE
        ):
            label = label2id.get("B-VENDOR", 0)

        # 5. Non-entity text
        else:
            label = label2id.get("O", 0)

        words.append(text)
        raw_boxes.append(bbox)
        labels.append(label)

    return words, raw_boxes, labels


def cache_ocr_dataset(image_dir="data/raw", output_json="data/processed_dataset.json"):
    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    cached_data = []

    for img_path in images:
        print(f"Processing OCR for: {img_path}")
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size

        res = OCR_ENGINE.predict(img_path)
        parsed_ocr = []

        for r in res:
            res_data = r.json["res"] if hasattr(r, "json") and "res" in r.json else r
            boxes = res_data.get("rec_polys", res_data.get("dt_polys", []))
            texts = res_data.get("rec_text", res_data.get("rec_texts", []))

            for b, t in zip(boxes, texts):
                box_list = b.tolist() if hasattr(b, "tolist") else b
                parsed_ocr.append({"box": box_list, "text": str(t)})

        if not parsed_ocr:
            continue

        words, raw_boxes, labels = auto_label_ocr(parsed_ocr, LABEL2ID)

        # Normalize bounding boxes
        norm_boxes = []
        for b in raw_boxes:
            nx1 = max(0, min(1000, int(1000 * (b[0] / w))))
            ny1 = max(0, min(1000, int(1000 * (b[1] / h))))
            nx2 = max(nx1, min(1000, int(1000 * (b[2] / w))))
            ny2 = max(ny1, min(1000, int(1000 * (b[3] / h))))
            norm_boxes.append([nx1, ny1, nx2, ny2])

        cached_data.append(
            {
                "image_path": img_path,
                "words": words,
                "boxes": norm_boxes,
                "labels": labels,
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cached_data, f, indent=4)

    print(f"\nSaved cached dataset to {output_json}")


if __name__ == "__main__":
    cache_ocr_dataset()
