from paddleocr import PaddleOCR


class OCRPipeline:
    def __init__(self, lang="en", device="cpu"):
        self.engine = PaddleOCR(
            use_textline_orientation=False,
            lang=lang,
            device=device,
            enable_mkldnn=False,
        )

    def extract_text(self, image_path):
        results = self.engine.predict(image_path)
        parsed_ocr = []

        for res in results:
            res_data = (
                res.json["res"] if hasattr(res, "json") and "res" in res.json else res
            )
            boxes = res_data.get("rec_polys", res_data.get("dt_polys", []))
            texts = res_data.get("rec_text", res_data.get("rec_texts", []))
            scores = res_data.get("rec_score", res_data.get("rec_scores", []))

            for box, text, score in zip(boxes, texts, scores):
                box_list = box.tolist() if hasattr(box, "tolist") else box
                parsed_ocr.append(
                    {
                        "box": box_list,
                        "text": str(text),
                        "confidence": round(float(score), 4),
                    }
                )

        return parsed_ocr
