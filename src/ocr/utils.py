def normalize_bbox(bbox, width, height):
    """
    Converts 4-corner polygon coords from PaddleOCR into
    LayoutLMv3's normalized [x_min, y_min, x_max, y_max] format on a 0-1000 grid.
    """
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    normalized_box = [
        int(1000 * (x_min / width)),
        int(1000 * (y_min / height)),
        int(1000 * (x_max / width)),
        int(1000 * (y_max / height)),
    ]

    return [max(0, min(1000, coord)) for coord in normalized_box]
