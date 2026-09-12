import re

AMOUNT_PATTERN = r"\b\d+(?:,\d{3})*(?:\.\d{2})\b"
TIN_PATTERN = r"\b\d{3}[-\s]\d{3}[-\s]\d{3}[-\s]\d{3,5}\b"


def extract_company(predictions):
    """Extract the company name from LayoutLM word predictions."""
    company_words = []

    for item in predictions:
        if item["label"] in {"B-COMPANY", "I-COMPANY"}:
            company_words.append(item["text"])

    return " ".join(company_words).strip()


def extract_date(predictions):
    """Extract the most likely transaction date from receipt text."""
    date_pattern = r"\b\d{2}[-/]\d{2}[-/]\d{4}" r"(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b"

    candidates = []

    for index, item in enumerate(predictions):
        text = item["text"]

        # Ignore dates explicitly associated with DATE ISSUED.
        if "DATE ISSUED" in text.upper():
            continue

        match = re.search(date_pattern, text)

        if match:
            candidate = match.group()

            # Prefer dates followed by a time.
            has_time = bool(re.search(r"\d{2}:\d{2}", candidate))

            candidates.append(
                {
                    "value": candidate,
                    "has_time": has_time,
                    "index": index,
                }
            )

    # A transaction date with a time is usually the strongest candidate.
    for candidate in candidates:
        if candidate["has_time"]:
            return candidate["value"]

    return candidates[0]["value"] if candidates else ""


def extract_tin(predictions):
    """Extract the most likely Philippine TIN from receipt text."""
    for item in predictions:
        text = item["text"]

        if "TIN" not in text.upper():
            continue

        match = re.search(TIN_PATTERN, text)

        if match:
            return match.group()

    # Also check nearby OCR words.
    for index, item in enumerate(predictions):
        if "TIN" not in item["text"].upper():
            continue

        nearby_text = " ".join(
            next_item["text"]
            for next_item in predictions[max(0, index - 1) : index + 3]
        )

        match = re.search(TIN_PATTERN, nearby_text)

        if match:
            return match.group()

    return ""


def extract_invoice_number(predictions):
    """Extract an invoice or sales invoice number using nearby labels."""
    invoice_patterns = [
        r"SALES\s+INVOICE",
        r"INVOICE\s+NO",
        r"SI\s*NO",
    ]

    # First look for a label and a number in the same OCR item.
    for index, item in enumerate(predictions):
        text = item["text"]

        for pattern in invoice_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                match = re.search(r"\b\d{4,}\b", text)

                if match:
                    return match.group()

    # Then check nearby OCR words in both directions.
    for index, item in enumerate(predictions):
        text = item["text"]

        for pattern in invoice_patterns:
            if not re.search(pattern, text, re.IGNORECASE):
                continue

            nearby_items = predictions[max(0, index - 2) : index + 3]

            for nearby_item in nearby_items:
                match = re.search(
                    r"\b\d{4,}\b",
                    nearby_item["text"],
                )

                if match:
                    return match.group()

    return ""


def extract_labeled_amount(predictions, label_pattern):
    """Find a numeric value associated with a nearby receipt label."""
    for index, item in enumerate(predictions):
        if not re.search(
            label_pattern,
            item["text"],
            re.IGNORECASE,
        ):
            continue

        for next_item in predictions[index + 1 : index + 3]:
            match = re.search(
                AMOUNT_PATTERN,
                next_item["text"],
            )

            if match:
                return float(match.group().replace(",", ""))

    return None


def extract_vatable_sales(predictions):
    """Extract Vatable Sales while avoiding zero-rated and exempt amounts."""
    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not re.fullmatch(r"Vatable(?:\s+Sales?)?", text, re.IGNORECASE):
            continue

        # The amount immediately following "Vatable" is the value we want.
        for next_item in predictions[index + 1 : index + 2]:
            match = re.search(
                AMOUNT_PATTERN,
                next_item["text"],
            )

            if match:
                return float(match.group().replace(",", ""))

    return None


def extract_vat_amount(predictions):
    """Extract the amount associated with VAT."""
    return extract_labeled_amount(
        predictions,
        r"[VU]AT\s*\(?12%\)?",
    )


def extract_total(predictions):
    """Extract the most likely receipt total using label priority."""
    # Prefer Total Sales because it is explicit on some Philippine receipts.
    total_labels = [
        r"TOTAL\s+SALES",
        r"^\s*TOTAL\s*$",
        r"AMOUNT\s+DUE",
    ]

    for label_pattern in total_labels:
        for index, item in enumerate(predictions):
            if not re.search(
                label_pattern,
                item["text"],
                re.IGNORECASE,
            ):
                continue

            for next_item in predictions[index + 1 : index + 3]:
                match = re.search(
                    AMOUNT_PATTERN,
                    next_item["text"],
                )

                if match:
                    return float(match.group().replace(",", ""))

    return None


def validate_vat(vatable_sales, vat_amount):
    """Check whether VAT is approximately 12% of Vatable Sales."""
    if vatable_sales is None or vat_amount is None:
        return False

    expected_vat = vatable_sales * 0.12

    return abs(expected_vat - vat_amount) <= 0.02


def extract_receipt_fields(predictions):
    """Extract and validate the main receipt fields."""
    vatable_sales = extract_vatable_sales(predictions)
    vat_amount = extract_vat_amount(predictions)

    return {
        "company": extract_company(predictions),
        "date": extract_date(predictions),
        "tin": extract_tin(predictions),
        "invoice_number": extract_invoice_number(predictions),
        "vatable_sales": vatable_sales,
        "vat_amount": vat_amount,
        "total": extract_total(predictions),
        "vat_valid": validate_vat(
            vatable_sales,
            vat_amount,
        ),
    }
