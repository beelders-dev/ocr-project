from PIL import Image

from src.preprocessing.image import resize_image_if_needed

image_path = "data/raw/receipt_2.jpg"

resized_path = resize_image_if_needed(image_path)

with Image.open(resized_path) as image:
    print(f"Output: {resized_path}")
    print(f"Size: {image.size}")
