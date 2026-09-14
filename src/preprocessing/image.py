from pathlib import Path
import tempfile

from PIL import Image

MAX_IMAGE_SIZE = 3300


def resize_image_if_needed(
    image_path: str,
    max_size: int = MAX_IMAGE_SIZE,
) -> str:
    """
    Resize an image only when its longest side exceeds max_size.

    Args:
        image_path: Path to the input image.
        max_size: Maximum allowed size for the longest image dimension.

    Returns:
        Path to the resized image, or the original path if no resize is needed.
    """
    path = Path(image_path)

    with Image.open(path) as image:
        width, height = image.size

        if max(width, height) <= max_size:
            return image_path

        scale = max_size / max(width, height)

        new_width = round(width * scale)
        new_height = round(height * scale)

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        with tempfile.NamedTemporaryFile(
            suffix=path.suffix,
            delete=False,
        ) as temp_file:
            output_path = Path(temp_file.name)
            resized.save(output_path)

    return str(output_path)
