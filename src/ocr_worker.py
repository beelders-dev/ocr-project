import sys

from paddleocr import PaddleOCR


def main():

    image_path = sys.argv[1]

    ocr_engine = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        device="cpu",
        enable_mkldnn=False,
    )

    result = ocr_engine.predict(image_path)

    for res in result:
        rec_texts = res.get("rec_texts", [])
        rec_boxes = res.get("rec_boxes", [])

        for text, box in zip(rec_texts, rec_boxes):
            text = str(text).strip()

            if not text:
                continue

            if hasattr(box, "tolist"):
                box = box.tolist()

            sys.stdout.buffer.write(f"{text}\t{box}\n".encode("utf-8"))

            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
