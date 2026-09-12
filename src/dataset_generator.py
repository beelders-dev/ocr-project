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
    "I-TOTAL",
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

    Examples:
        "RM 69.20" -> "69.20"
        "69.20"    -> "69.20"
        "69.2"     -> "69.20"
        "NETT TOTAL: $8.20" -> "8.20"

    Returns None when the text does not look like a monetary value.
    """
    text = str(text).strip().upper().replace(",", "")

    # Look for a number that has a decimal portion.
    decimal_match = re.search(
        r"\d+\.\d{1,2}",
        text,
    )

    if decimal_match:
        try:
            return f"{float(decimal_match.group(0)):.2f}"
        except ValueError:
            return None

    # Allow plain integers only when the entire OCR word is numeric
    # or contains a currency symbol/code.
    cleaned = re.sub(r"^(RM|MYR|\$)\s*", "", text).strip()

    if re.fullmatch(r"\d+", cleaned):
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            return None

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

    OCR may contain additional information such as time,
    and the date format may differ from the ground truth.
    """

    target = normalize_text(entity_value)

    if not target:
        return []

    # First try the existing exact normalized match.
    for index, word in enumerate(words):
        normalized_word = normalize_text(word)

        if target in normalized_word:
            return [index]

    # Extract a normalized date as (year, month, day).
    def parse_date(text):
        text = str(text).strip().upper()

        # YYYYMMDD
        match = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", text)
        if match:
            return (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

        # YYYY-MM-DD
        match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
        if match:
            return (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

        # DD/MM/YYYY or DD-MM-YYYY
        match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", text)
        if match:
            return (
                int(match.group(3)),
                int(match.group(2)),
                int(match.group(1)),
            )

        # DD/MM/YY or DD-MM-YY
        match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2})\b", text)
        if match:
            return (
                2000 + int(match.group(3)),
                int(match.group(2)),
                int(match.group(1)),
            )

        # Month-name formats:
        # 30 DEC 17
        # OCT 3, 2016
        month_names = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }

        month_pattern = (
            r"\b("
            + "|".join(month_names.keys())
            + r")\s+(\d{1,2})(?:,)?\s+(20\d{2}|\d{2})\b"
        )

        match = re.search(month_pattern, text)

        if match:
            year = int(match.group(3))

            if year < 100:
                year += 2000

            return (
                year,
                month_names[match.group(1)],
                int(match.group(2)),
            )

        # Reverse month format:
        # 3 OCT 2016
        match = re.search(
            r"\b(\d{1,2})\s+("
            + "|".join(month_names.keys())
            + r")(?:,)?\s+(20\d{2}|\d{2})\b",
            text,
        )

        if match:
            year = int(match.group(3))

            if year < 100:
                year += 2000

            return (
                year,
                month_names[match.group(2)],
                int(match.group(1)),
            )

        return None

    target_date = parse_date(entity_value)

    if target_date is None:
        return []

    # Try to find the same date in OCR, even when formatting differs.
    for index, word in enumerate(words):
        word_date = parse_date(word)

        if word_date == target_date:
            return [index]

    # Allow partial OCR dates such as:
    # Ground truth: "OCT 3, 2016"
    # OCR:          "OCT 3"
    target_month = target_date[1]
    target_day = target_date[2]

    month_names = {
        1: "JAN",
        2: "FEB",
        3: "MAR",
        4: "APR",
        5: "MAY",
        6: "JUN",
        7: "JUL",
        8: "AUG",
        9: "SEP",
        10: "OCT",
        11: "NOV",
        12: "DEC",
    }

    month_name = month_names[target_month]

    for index, word in enumerate(words):
        normalized_word = normalize_text(word)

        if not normalized_word:
            continue

        if re.search(
            rf"\b{month_name}\s*{target_day}\b",
            normalized_word,
        ):
            return [index]

    return []


def find_total_word(words, boxes, entity_value):
    """Find the OCR word that corresponds to the receipt total.

    Strategy:
    1. Find meaningful total-related labels.
    2. Ignore labels that only contain "TOTAL" as part of another
       word, such as "SUBTOTAL".
    3. If SROIE provides an expected total, prefer an exact numeric
       match near a strong total label.
    4. If no label is available, use the exact numeric match.
    5. If the SROIE total is missing, use a numeric value directly
       attached to a strong total label.
    """

    normalized_target = normalize_number(entity_value)

    target_number = None
    if normalized_target is not None:
        target_number = float(normalized_target)

    # Higher priority means stronger evidence that the label
    # identifies the final receipt total.
    total_keywords = {
        "AMOUNT TO BE PAID": 100,
        "GRAND TOTAL": 95,
        "TOTAL AMOUNT": 90,
        "TOTAL SALES": 85,
        "TOTAL ROUNDED": 80,
        "NETT TOTAL": 75,
        "TOTAL DUE": 70,
        "TOTAL": 50,
    }

    # These words should never be treated as total labels.
    excluded_keywords = {
        "SUBTOTAL",
        "CASH",
        "CASH RECEIVED",
        "CASH TENDERED",
        "CHANGE",
        "DISCOUNT",
        "ROUNDING",
        "ROUNDING ADJUSTMENT",
        "TAX",
        "VAT",
        "GST",
    }

    label_candidates = []

    # ---------------------------------------------------------
    # 1. Find meaningful total labels.
    # ---------------------------------------------------------

    for index, word in enumerate(words):
        normalized_word = normalize_text(word)

        if not normalized_word:
            continue

        # Explicitly reject known non-total labels.
        if any(
            normalize_text(excluded) in normalized_word
            for excluded in excluded_keywords
        ):
            continue

        best_priority = None
        best_keyword = None

        for keyword, priority in total_keywords.items():
            normalized_keyword = normalize_text(keyword)

            # For the generic TOTAL keyword, require the entire
            # OCR word to be TOTAL. This prevents SUBTOTAL from
            # becoming a total label.
            if keyword == "TOTAL":
                if normalized_word == "TOTAL":
                    if best_priority is None or priority > best_priority:
                        best_priority = priority
                        best_keyword = keyword
                continue

            # Longer/more specific labels can still appear inside
            # OCR text such as "TOTALSALESINCLUSIVEGST".
            if normalized_keyword in normalized_word:
                if best_priority is None or priority > best_priority:
                    best_priority = priority
                    best_keyword = keyword

        if best_priority is not None:
            label_candidates.append(
                {
                    "index": index,
                    "priority": best_priority,
                    "keyword": best_keyword,
                }
            )

    # ---------------------------------------------------------
    # 2. We have an expected SROIE total and total labels.
    #
    # Find the exact expected amount nearest to the strongest
    # total label.
    # ---------------------------------------------------------

    if target_number is not None and label_candidates:
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

                same_line = vertical_distance <= 40

                value_difference = abs(number - target_number)

                exact_match = value_difference < 0.01

                if x1 >= label_x2:
                    horizontal_distance = x1 - label_x2
                elif x2 <= label_x1:
                    horizontal_distance = label_x1 - x2
                else:
                    horizontal_distance = 0

                score = 0.0

                # The expected SROIE total is strong evidence.
                if exact_match:
                    score += 1000

                # A value on the same line as the label is strong
                # spatial evidence.
                if same_line:
                    score += 500

                # Prefer semantically stronger labels.
                score += label["priority"]

                # Prefer nearby values.
                score -= horizontal_distance * 0.1
                score -= vertical_distance * 2

                candidates.append(
                    {
                        "index": index,
                        "score": score,
                        "label": label["keyword"],
                    }
                )

        if candidates:
            candidates.sort(
                key=lambda candidate: candidate["score"],
                reverse=True,
            )

            best = candidates[0]

            print(
                "TOTAL DEBUG:",
                entity_value,
                "=>",
                words[best["index"]],
                f"(label: {best['label']})",
            )

            return [best["index"]]

    # ---------------------------------------------------------
    # 3. No useful total label.
    #
    # If SROIE gives us the expected total, find exact numeric
    # matches. When multiple occurrences exist, prefer the last
    # occurrence as a fallback because receipt totals often appear
    # again near the bottom of the receipt.
    # ---------------------------------------------------------

    if target_number is not None:
        exact_matches = []

        for index, word in enumerate(words):
            normalized_number = normalize_number(word)

            if normalized_number is None:
                continue

            number = float(normalized_number)

            if abs(number - target_number) < 0.01:
                exact_matches.append(index)

        if exact_matches:
            return [exact_matches[-1]]

    # ---------------------------------------------------------
    # 4. SROIE total is missing.
    #
    # Look for a numeric value directly associated with a strong
    # total label.
    # ---------------------------------------------------------

    if label_candidates:
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

                box = boxes[index]
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) / 2

                vertical_distance = abs(center_y - label_center_y)

                same_line = vertical_distance <= 40

                if x1 >= label_x2:
                    horizontal_distance = x1 - label_x2
                elif x2 <= label_x1:
                    horizontal_distance = label_x1 - x2
                else:
                    horizontal_distance = 0

                score = label["priority"]

                if same_line:
                    score += 500

                score -= horizontal_distance * 0.1
                score -= vertical_distance * 2

                candidates.append(
                    {
                        "index": index,
                        "score": score,
                        "label": label["keyword"],
                    }
                )

        if candidates:
            candidates.sort(
                key=lambda candidate: candidate["score"],
                reverse=True,
            )

            best = candidates[0]

            return [best["index"]]

    return []


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
    Find OCR words that correspond to the SROIE address entity.

    Strategy:
    1. Match complete OCR words against portions of the address.
    2. Handle common address-number patterns such as:
       - NO.53
       - NO.2&4
       - NO. 343
       - 27
       - postal codes
    3. Prefer consecutive OCR words that occur in the same
       address/header region.
    4. Reject obvious company/header text.
    5. Never invent labels for text that does not exist in OCR.
    """

    if not entity_value:
        return []

    # ---------------------------------------------------------
    # Normalize the complete address.
    # ---------------------------------------------------------

    target = normalize_text(entity_value)

    if not target:
        return []

    normalized_words = [normalize_text(word) for word in words]

    # ---------------------------------------------------------
    # 1. Exact consecutive-word matching.
    #
    # This is the safest case.
    # ---------------------------------------------------------

    exact_match = find_exact_sequence(
        words,
        entity_value,
    )

    if exact_match:
        return exact_match

    # ---------------------------------------------------------
    # 2. Build useful address fragments.
    #
    # We compare OCR words against the normalized address
    # rather than comparing them against the entire entity.
    # ---------------------------------------------------------

    address_words = [normalize_text(word) for word in str(entity_value).split()]

    address_words = [word for word in address_words if word]

    # ---------------------------------------------------------
    # 3. Reject obvious company/header words.
    #
    # These caused false positives such as:
    # BOOK TA .K(TAMAN DAYA) SDN BND
    # MR D.T.Y. (JOHOR) SDN BHD
    # GERBANG ALAF RESTAURANTS SDN BHD
    # ---------------------------------------------------------

    company_patterns = {
        "SDNBHD",
        "SDNBHD.",
        "ENTERPRISE",
        "RESTAURANTS",
        "RESTAURANT",
        "TRADING",
        "HOLDINGS",
        "CORPORATION",
        "CORP",
        "LIMITED",
        "LTD",
        "BERHAD",
        "BHD",
        "PTE",
        "INC",
    }

    def looks_like_company(word_index):
        text = normalize_text(words[word_index])

        for pattern in company_patterns:
            if pattern in text:
                return True

        return False

    # ---------------------------------------------------------
    # 4. Find address-number candidates.
    #
    # These are important because SROIE often has:
    #
    # NO.53
    # NO.2&4
    # NO. 343
    # 27
    # LOT 1851-A
    # 81100
    #
    # But avoid monetary values.
    # ---------------------------------------------------------

    def is_address_number(word):
        raw = str(word).strip().upper()
        normalized = normalize_text(raw)

        if not normalized:
            return False

        # Money-like values should not become addresses.
        if re.fullmatch(
            r"\d+(?:[.,]\d{1,2})?",
            raw.replace(" ", ""),
        ):
            # Postal codes are five digits in these receipts.
            if re.fullmatch(r"\d{5}", normalized):
                return True

            # Small plain integers can be house numbers.
            if re.fullmatch(r"\d{1,4}", normalized):
                return True

            return False

        # NO.53 / NO.2&4 / NO. 343
        if re.search(r"\bNO", normalized):
            if re.search(r"\d", normalized):
                return True

        # LOT 1851-A / LOT 123
        if normalized.startswith("LOT") and re.search(
            r"\d",
            normalized,
        ):
            return True

        return False

    # ---------------------------------------------------------
    # 5. Match OCR words against address fragments.
    #
    # Exact substring matches are preferred.
    # ---------------------------------------------------------

    candidates = []

    for index, word in enumerate(words):

        normalized_word = normalized_words[index]

        if not normalized_word:
            continue

        # Never use obvious company words as address anchors.
        if looks_like_company(index):
            continue

        # Skip very short generic words.
        if len(normalized_word) < 2:
            continue

        # Avoid OCR text that is clearly not address-like.
        if normalized_word in {
            "TEL",
            "FAX",
            "EMAIL",
            "GST",
            "TAX",
            "CASH",
            "CHANGE",
            "SUBTOTAL",
            "TOTAL",
        }:
            continue

        best_score = 0.0

        combined = ""

        for address_word in address_words:
            combined += address_word

            if len(combined) < 3:
                continue

            if combined == normalized_word:

                best_score = 1.0
                break

        for address_word in address_words:

            if not address_word:
                continue

            # Exact normalized match.
            if normalized_word == address_word:
                score = 1.0

            # OCR word can contain the address fragment.
            elif len(address_word) >= 4 and address_word in normalized_word:
                score = 0.95

            elif len(normalized_word) >= 4 and normalized_word in address_word:
                score = 0.90

            else:
                score = 0.0

            best_score = max(
                best_score,
                score,
            )

        # -----------------------------------------------------
        # Numeric address components.
        # -----------------------------------------------------

        if is_address_number(word):

            if normalized_word in target:
                best_score = max(
                    best_score,
                    0.95,
                )

        if best_score > 0:

            candidates.append(
                {
                    "index": index,
                    "score": best_score,
                }
            )

    if not candidates:
        return []

    # ---------------------------------------------------------
    # 6. Sort candidates by OCR position.
    # ---------------------------------------------------------

    candidates.sort(key=lambda item: item["index"])

    candidate_indexes = {item["index"] for item in candidates}

    # ---------------------------------------------------------
    # 7. Find the strongest address run.
    #
    # Address OCR is normally represented as one or several
    # consecutive lines. We therefore prefer candidates that
    # occur close together instead of isolated matches.
    # ---------------------------------------------------------

    runs = []

    current_run = []

    for candidate in candidates:

        index = candidate["index"]

        if not current_run:
            current_run = [candidate]
            continue

        previous_index = current_run[-1]["index"]

        # Consecutive OCR words.
        if index == previous_index + 1:
            current_run.append(candidate)
            continue

        runs.append(current_run)
        current_run = [candidate]

    if current_run:
        runs.append(current_run)

    # ---------------------------------------------------------
    # 8. Score each run.
    #
    # Consecutive address words are much stronger than a single
    # fuzzy match such as "TAMAN" inside a company name.
    # ---------------------------------------------------------

    scored_runs = []

    for run in runs:

        indexes = [item["index"] for item in run]

        score = sum(item["score"] for item in run)

        # Strong bonus for multiple consecutive matches.
        if len(run) >= 2:
            score += 1.0

        if len(run) >= 3:
            score += 1.0

        # Bonus when the run contains an address number.
        if any(is_address_number(words[index]) for index in indexes):
            score += 1.5

        scored_runs.append(
            {
                "indexes": indexes,
                "score": score,
            }
        )

    scored_runs.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best_run = scored_runs[0]

    # ---------------------------------------------------------
    # 9. Reject weak isolated matches.
    #
    # This prevents things like:
    #
    # GERBANG ALAF RESTAURANTS SDN BHD
    #
    # from becoming B-ADDRESS merely because it contains
    # "JOHOR" or another generic location word.
    # ---------------------------------------------------------

    if len(best_run["indexes"]) == 1:

        index = best_run["indexes"][0]

        word = words[index]
        normalized_word = normalized_words[index]

        # A lone location word is too weak.
        if normalized_word in {
            "JOHOR",
            "SELANGOR",
            "BAHRU",
            "PETALING",
            "MALAYSIA",
        }:
            return []

        # A lone non-numeric word is also too weak unless it
        # is a strong address marker.
        strong_markers = {
            "LEVEL",
            "LOT",
            "NO",
            "JALAN",
            "JLN",
            "ROAD",
            "STREET",
            "TAMAN",
            "BANDAR",
            "KAWASAN",
            "PERSIARAN",
            "BLOCK",
            "UNIT",
            "BANGUNAN",
        }

        if normalized_word not in strong_markers:
            if not is_address_number(word):
                if best_run["score"] < 0.90:
                    return []

    # ---------------------------------------------------------
    # 10. Expand only to nearby candidates.
    #
    # Do NOT spatially expand from arbitrary words.
    # This prevents a false company match from pulling in
    # unrelated receipt text.
    # ---------------------------------------------------------

    selected = set(best_run["indexes"])

    best_start = min(selected)
    best_end = max(selected)

    for candidate in candidates:

        index = candidate["index"]

        if index in selected:
            continue

        # Only include nearby OCR words.
        if index < best_start - 1:
            continue

        if index > best_end + 1:
            continue

        # Require a reasonably strong match.
        if candidate["score"] < 0.90:
            continue

        if looks_like_company(index):
            continue

        selected.add(index)

        print(
            "ADDRESS DEBUG:",
            entity_value,
            "=>",
            [words[i] for i in sorted(selected)],
        )
    return sorted(selected)


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

        # OCR may combine the company name with
        # registration numbers or other company information.
        if target in word:
            return [index]

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

    # Total can still be detected even when SROIE
    # does not provide a ground-truth total.
    if entity_type == "total":
        return find_total_word(
            words,
            boxes,
            entity_value,
        )

    if not entity_value:
        return []

    if entity_type == "date":
        return find_date_word(
            words,
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

        indexes = find_entity_words(
            words,
            boxes,
            entity_value,
            entity_name,
        )

        if not indexes:
            continue

        assigned_count = 0

        for word_index in indexes:

            # Don't allow a later entity to overwrite
            # an already assigned entity.
            if ner_tags[word_index] != LABEL2ID["O"]:
                continue

            prefix = "B-" if assigned_count == 0 else "I-"

            label = f"{prefix}{entity_label}"

            ner_tags[word_index] = LABEL2ID[label]

            assigned_count += 1

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
        # if idx >= 50:
        #     break

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

        image = sample.get("image")

        # ---------------------------------------------------------
        # DEBUG RECEIPT 25
        # ---------------------------------------------------------
        if idx == 38:
            print("\n--- RECEIPT 25 DEBUG ---")

            print("Words:")
            for index, word in enumerate(words):
                print(index, repr(word))

            print("\nEntities:")
            print(entities)

            print("\nDetected total:")
            total_indexes = find_total_word(
                words,
                bboxes,
                entities.get("total", ""),
            )

            print(total_indexes)

            for index in total_indexes:
                print(
                    "Selected word:",
                    index,
                    repr(words[index]),
                )

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

        detected = {}

        for word_index, (word, tag_id) in enumerate(zip(words, ner_tags)):
            if tag_id == LABEL2ID["O"]:
                continue

            label = LABEL_LIST[tag_id]
            entity_type = label[2:]  # Remove B- / I-

            detected.setdefault(entity_type, [])
            detected[entity_type].append(word)

        for entity_type in ["COMPANY", "ADDRESS", "DATE", "TOTAL"]:
            value = " ".join(detected.get(entity_type, []))
            print(f"  {entity_type:<8}: {value or '[NOT FOUND]'}")

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
