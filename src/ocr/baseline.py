import json
from paddleocr import PaddleOCR

# Force CPU execution to bypass Windows C++ DLL dependencies
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

image_path = "data/raw/sample_receipt.jpg"

# Execute DBNet (detection) and CRNN (recognition) on CPU
result = ocr.ocr(image_path, cls=False)

parsed_regions = []
if result and result[0]:
    for line in result[0]:
        parsed_regions.append(
            {
                "box": line[0],  # Bounding box coordinates
                "text": line[1][0],  # Recognized text string
                "confidence": round(line[1][1], 4),  # Prediction score
            }
        )

print(json.dumps(parsed_regions, indent=2))
