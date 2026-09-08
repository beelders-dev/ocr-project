import json
import os
import re

from difflib import SequenceMatcher

from datasets import load_dataset

LABEL_LIST = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


def sanitize_box(box):
    """Clamp a bounding box to the LayoutLMv3 [0, 1000] range."""
    if not box or len(box) != 4:
        return [0, 0, 1, 1]

    x1, y1, x2, y2 = [max(0, min(1000, int(coord))) for coord in box]

    if x2 <= x1:
        x2 = min(1000, x1 + 1)

    if y2 <= y1:
        y2 = min(1000, y1 + 1)

    return [x1, y1, x2, y2]


def normalize_text(text):
    """
    Normalize text for comparison.

    Example:
        "09 JUN 2018" -> "09JUN2018"
        "09/06/2018"  -> "09062018"
    """
    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_number(text):
    """
    Normalize a monetary value.

    Example:
        "RM 69.20" -> "69.20"
        "69.20"    -> "69.20"
        "69.2"     -> "69.20"
    """
    text = str(text).replace(",", "")

    match = re.search(
        r"\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:
        return f"{float(match.group(0)):.2f}"
    except ValueError:
        return None


def box_center(box):
    """Return the center point of a normalized bounding box."""
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def vertical_distance(box1, box2):
    """
    Calculate vertical distance between two bounding boxes.
    """
    _, y1 = box_center(box1)
    _, y2 = box_center(box2)

    return abs(y1 - y2)


def horizontal_distance(box1, box2):
    """
    Calculate horizontal distance between two bounding boxes.
    """
    x1, _ = box_center(box1)
    x2, _ = box_center(box2)

    return abs(x1 - x2)


def find_date_word(words, entity_value):
    """
    Find the OCR word containing the expected date.

    OCR may contain additional information such as time.
    """
    target = normalize_text(entity_value)

    if not target:
        return []

    for index, word in enumerate(words):
        normalized_word = normalize_text(word)

        if target in normalized_word:
            return [index]

    return []


def find_total_word(words, boxes, entity_value):
    """
    Find the OCR word that corresponds to the receipt total.

    The function returns a list of OCR indexes because
    create_ner_tags() expects a list.
    """

    normalized_target = normalize_number(entity_value)

    if normalized_target is None:
        return []

    target_number = float(normalized_target)

    total_keywords = [
        "TOTAL ROUNDED",
        "GRAND TOTAL",
        "TOTAL AMT",
        "TOTAL AMOUNT",
        "TOTAL SALES",
        "TOTAL",
    ]

    label_candidates = []

    # Find OCR words that look like total labels.
    for index, word in enumerate(words):
        normalized_word = normalize_text(word)

        for priority, keyword in enumerate(total_keywords):
            normalized_keyword = normalize_text(keyword)

            if normalized_keyword in normalized_word:
                label_candidates.append(
                    {
                        "index": index,
                        "priority": priority,
                    }
                )
                break

    if not label_candidates:
        return []

    candidates = []

    for label in label_candidates:
        label_index = label["index"]
        label_box = boxes[label_index]

        label_x1, label_y1, label_x2, label_y2 = label_box

        label_center_y = (label_y1 + label_y2) / 2

        for index, word in enumerate(words):

            if index == label_index:
                continue

            normalized_number = normalize_number(word)

            if normalized_number is None:
                continue

            number = float(normalized_number)

            box = boxes[index]

            x1, y1, x2, y2 = box

            center_y = (y1 + y2) / 2

            vertical_distance = abs(center_y - label_center_y)

            # Is the amount on the same horizontal line?
            same_line = vertical_distance <= 40

            # Does the OCR value match the SROIE entity?
            value_difference = abs(number - target_number)

            exact_match = value_difference < 0.01

            # Horizontal distance from the total label.
            if x1 >= label_x2:
                horizontal_distance = x1 - label_x2
            elif x2 <= label_x1:
                horizontal_distance = label_x1 - x2
            else:
                horizontal_distance = 0

            score = 0

            # Exact amount is extremely important.
            if exact_match:
                score += 1000

            # Same-line values are strongly preferred.
            if same_line:
                score += 500

            # Prefer more specific total labels.
            score += (len(total_keywords) - label["priority"]) * 100

            # Prefer values horizontally close to the label.
            score -= horizontal_distance * 0.1

            # Penalize vertical distance.
            score -= vertical_distance * 2

            candidates.append(
                {
                    "index": index,
                    "score": score,
                }
            )

    if not candidates:
        return []

    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    best = candidates[0]

    return [best["index"]]


def find_exact_sequence(words, entity_value):
    """
    Find an entity by combining consecutive OCR words.

    Punctuation and spaces are ignored during comparison.
    """
    target = normalize_text(entity_value)

    if not target:
        return []

    normalized_words = [normalize_text(word) for word in words]

    for start in range(len(words)):
        combined = ""

        for end in range(start, len(words)):
            combined += normalized_words[end]

            if combined == target:
                return list(range(start, end + 1))

            if len(combined) > len(target):
                break

    return []


def find_address_words(words, boxes, entity_value):
    """
    Find address words conservatively.

    SROIE entity addresses may contain text that is missing
    from the OCR words, so we only label OCR words that we
    can match with reasonable confidence.

    We intentionally avoid aggressive fuzzy matching because
    false address labels are worse than missing labels.
    """

    target = normalize_text(entity_value)

    if not target:
        return []

    # ---------------------------------------------------------
    # 1. Exact consecutive-word matching.
    # ---------------------------------------------------------

    exact_match = find_exact_sequence(
        words,
        entity_value,
    )

    if exact_match:
        return exact_match

    # ---------------------------------------------------------
    # 2. Build normalized target words.
    # ---------------------------------------------------------

    target_words = [normalize_text(word) for word in str(entity_value).split()]

    target_words = [word for word in target_words if len(word) >= 3]

    if not target_words:
        return []

    # ---------------------------------------------------------
    # Address words that are useful signals.
    # ---------------------------------------------------------

    address_keywords = {
        "JALAN",
        "JLN",
        "ROAD",
        "STREET",
        "TAMAN",
        "BANDAR",
        "KAWASAN",
        "PERINDUSTRIAN",
        "PERSIARAN",
        "PETALING",
        "JOHOR",
        "BAHRU",
        "SELANGOR",
        "PENGERANG",
        "KEMBANGAN",
        "DAMANSARA",
        "UPTOWN",
        "LEVEL",
        "NO",
        "LOT",
        "BLOCK",
        "UNIT",
        "BANGUNAN",
    }

    candidates = []

    for index, word in enumerate(words):
        normalized_word = normalize_text(word)

        if len(normalized_word) < 3:
            continue

        # Never treat decimal amounts as address words.
        if "." in str(word):
            number = normalize_number(word)

            if number is not None:
                continue

        best_score = 0.0
        best_target = None

        for target_word in target_words:
            score = SequenceMatcher(
                None,
                normalized_word,
                target_word,
            ).ratio()

            if score > best_score:
                best_score = score
                best_target = target_word

        if best_score < 0.85:
            continue

        keyword_bonus = 0.0

        for keyword in address_keywords:
            if keyword in normalized_word:
                keyword_bonus = 0.10
                break

        candidates.append(
            {
                "index": index,
                "score": best_score + keyword_bonus,
                "target": best_target,
            }
        )

    if not candidates:
        return []

    # ---------------------------------------------------------
    # 3. Only keep strong matches.
    # ---------------------------------------------------------

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Don't start an address from a generic location word.
    generic_location_words = {
        "JOHOR",
        "SELANGOR",
        "BAHRU",
        "PETALING",
        "MALAYSIA",
    }

    valid_starts = [
        candidate
        for candidate in candidates
        if candidate["target"] not in generic_location_words
    ]

    if not valid_starts:
        return []

    strongest = valid_starts[0]

    selected = [strongest["index"]]

    # ---------------------------------------------------------
    # 4. Add other strong candidates that are spatially close.
    # ---------------------------------------------------------

    for candidate in candidates[1:]:
        index = candidate["index"]

        if index == strongest["index"]:
            continue

        # Don't accept weak fuzzy matches.
        if candidate["score"] < 0.85:
            continue

        close_to_existing = False

        for selected_index in selected:
            y_distance = vertical_distance(
                boxes[selected_index],
                boxes[index],
            )

            x_distance = horizontal_distance(
                boxes[selected_index],
                boxes[index],
            )

            # Same line / nearby line.
            if y_distance <= 100 and x_distance <= 500:
                close_to_existing = True
                break

        if close_to_existing:
            selected.append(index)

    selected.sort()

    return selected


def find_company_words(words, entity_value):
    """
    Find company words.
    """
    exact_match = find_exact_sequence(
        words,
        entity_value,
    )

    if exact_match:
        return exact_match

    target = normalize_text(entity_value)

    if not target:
        return []

    normalized_words = [normalize_text(word) for word in words]

    best_index = None
    best_score = 0.0

    for index, word in enumerate(normalized_words):
        if not word:
            continue

        score = SequenceMatcher(
            None,
            word,
            target,
        ).ratio()

        if score > best_score:
            best_score = score
            best_index = index

    if best_index is not None and best_score >= 0.60:
        return [best_index]

    return []


def find_entity_words(
    words,
    boxes,
    entity_value,
    entity_type,
):
    """
    Find OCR words corresponding to an entity.
    """
    if not entity_value:
        return []

    if entity_type == "date":
        return find_date_word(
            words,
            entity_value,
        )

    if entity_type == "total":
        return find_total_word(
            words,
            boxes,
            entity_value,
        )

    if entity_type == "company":
        return find_company_words(
            words,
            entity_value,
        )

    if entity_type == "address":
        return find_address_words(
            words,
            boxes,
            entity_value,
        )

    return []


def create_ner_tags(words, boxes, entities):
    """
    Convert document-level SROIE entities into BIO word labels.
    """
    ner_tags = [LABEL2ID["O"]] * len(words)

    entity_labels = {
        "company": "COMPANY",
        "date": "DATE",
        "address": "ADDRESS",
        "total": "TOTAL",
    }

    for entity_name, entity_label in entity_labels.items():
        entity_value = entities.get(
            entity_name,
            "",
        )

        if not entity_value:
            continue

        indexes = find_entity_words(
            words,
            boxes,
            entity_value,
            entity_name,
        )

        if not indexes:
            continue

        for position, word_index in enumerate(indexes):
            prefix = "B-" if position == 0 else "I-"

            label = f"{prefix}{entity_label}"

            ner_tags[word_index] = LABEL2ID[label]

    return ner_tags


def download_and_format_sroie():
    print("Downloading SROIE dataset " "from Hugging Face Hub...")

    raw_dataset = load_dataset(
        "jsdnrs/ICDAR2019-SROIE",
        split="train",
    )

    processed_samples = []

    print("Formatting SROIE annotations " "for LayoutLMv3...")

    for idx, sample in enumerate(raw_dataset):

        # Keep this at 10 while testing.
        if idx >= 10:
            break

        words = sample.get(
            "words",
            [],
        )

        bboxes = sample.get(
            "bboxes",
            [],
        )

        entities = sample.get(
            "entities",
            {},
        )

        print(f"\nADDRESS ENTITY {idx}:", entities.get("address", ""))

        print("OCR WORDS:")

        for word_index, word in enumerate(words):
            print(f"  {word_index}: {word}")

        image = sample.get("image")

        if idx == 0:
            print("\n--- DEBUG SAMPLE ---")
            print("Words:", words[:10])
            print("Entities:", entities)
            print("Entity type:", type(entities))
            print("Image type:", type(image))

        if not words or not bboxes:
            continue

        if len(words) != len(bboxes):
            print(
                f"Skipping sample {idx}: "
                f"{len(words)} words but "
                f"{len(bboxes)} boxes."
            )
            continue

        clean_boxes = [sanitize_box(box) for box in bboxes]

        ner_tags = create_ner_tags(
            words,
            clean_boxes,
            entities,
        )

        print(f"\nReceipt {idx}")

        for word_index, (
            word,
            tag_id,
        ) in enumerate(zip(words, ner_tags)):
            if tag_id != LABEL2ID["O"]:
                label = LABEL_LIST[tag_id]

                print(f"  {word_index}: " f"{label:<12} -> {word}")

        sample_dict = {
            "id": str(idx),
            # Store the actual image directly.
            "image": image,
            "words": words,
            # SROIE boxes are already normalized
            # to the [0, 1000] LayoutLM format.
            "boxes": clean_boxes,
            "ner_tags": ner_tags,
        }

        processed_samples.append(sample_dict)

    os.makedirs(
        "data",
        exist_ok=True,
    )

    output_path = "data/processed_dataset.json"

    # JSON cannot store PIL images.
    #
    # Therefore, save the images separately.
    image_dir = "data/images"

    os.makedirs(
        image_dir,
        exist_ok=True,
    )

    for sample in processed_samples:
        image = sample.pop("image")

        image_path = os.path.join(
            image_dir,
            f"sroie_{sample['id']}.png",
        )

        image.save(image_path)

        sample["image_path"] = image_path

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            processed_samples,
            file,
            indent=2,
        )

    print(
        f"\nSuccessfully formatted "
        f"{len(processed_samples)} receipts "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    download_and_format_sroie()
