from paddleocr import PaddleOCR

OCR_ENGINE = PaddleOCR(
    use_textline_orientation=False, lang="en", device="cpu", enable_mkldnn=False
)
