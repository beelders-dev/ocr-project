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
    date_patterns = [
        # 09/12/2026, 09-12-2026, optionally with time
        r"\b\d{2}[-/]\d{2}[-/]\d{4}" r"(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b",
        # 09-03-26, optionally with time and AM/PM
        r"\b\d{2}[-/]\d{2}[-/]\d{2}" r"(?:\s+\d{2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)?\b",
        # 30 Aug 26 18:01:52
        r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}" r"(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b",
    ]

    candidates = []

    for index, item in enumerate(predictions):
        text = item["text"]

        if "DATE ISSUED" in text.upper():
            continue

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)

            if not match:
                continue

            candidate = match.group()

            has_time = bool(re.search(r"\d{1,2}:\d{2}", candidate))

            candidates.append(
                {
                    "value": candidate,
                    "has_time": has_time,
                    "index": index,
                }
            )

            break

    # A transaction timestamp is usually the strongest candidate.
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
        r"SALES\s+INVOICE(?:\s+(?:NO|NUMBER))?",
        r"INVOICE\s*[#№]?\s*(?:NO\.?|NUMBER)?",
        r"INV(?:OICE)?\s*[#№]?\s*(?:NO\.?|NUMBER)?",
        r"SI\s*[#№]?\s*(?:NO\.?|NUMBER)?",
    ]

    number_pattern = r"\b\d{4,}\b"

    # First look for a label and number in the same OCR item.
    for index, item in enumerate(predictions):
        text = item["text"]

        if not any(
            re.search(pattern, text, re.IGNORECASE) for pattern in invoice_patterns
        ):
            continue

        # Remove obvious non-invoice numbers such as dates.
        candidates = re.findall(number_pattern, text)

        for candidate in candidates:
            if len(candidate) >= 4:
                return candidate

    # Then check nearby OCR words.
    for index, item in enumerate(predictions):
        text = item["text"]

        if not any(
            re.search(pattern, text, re.IGNORECASE) for pattern in invoice_patterns
        ):
            continue

        nearby_items = predictions[max(0, index - 2) : index + 4]

        for nearby_item in nearby_items:
            candidate = re.search(
                number_pattern,
                nearby_item["text"],
            )

            if candidate:
                return candidate.group()

    return ""


def extract_amount(text):
    """Extract a receipt amount from text, including common currency prefixes."""
    amount_pattern = r"(?<!\d)\d+(?:,\d{3})*(?:\.\d{2})(?!\d)"

    match = re.search(amount_pattern, text)

    if not match:
        return None

    return float(match.group().replace(",", ""))


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
    """Extract Vatable Sales from an amount associated with its label."""
    vatable_patterns = [
        r"^/?v?atable(?:\s+sales?)?$",
        r"^vat\s+sales$",
        r"^sales\s+excl\.?\s*(?:of\s+)?vat$",
        r"^net\s+sales$",
    ]

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not any(
            re.fullmatch(pattern, text, re.IGNORECASE) for pattern in vatable_patterns
        ):
            continue

        # Prefer amounts appearing after the label.
        for nearby_item in predictions[index + 1 : index + 4]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

        # Fall back to amounts before the label when OCR order is reversed.
        for nearby_item in predictions[max(0, index - 2) : index]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

    return None


def extract_vat_amount(predictions):
    """Extract VAT Amount from an amount associated with its label."""
    vat_patterns = [
        r"^/?vat$",
        r"^/?vat\s+amount$",
        r"^/?vat\s*(?:[-:]|\(12%\)|12%)?$",
        r"^12%\s+vat$",
        r"^plus\s+vat\s+amount$",
    ]

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not any(
            re.fullmatch(pattern, text, re.IGNORECASE) for pattern in vat_patterns
        ):
            continue

        # Prefer amounts appearing after the VAT label.
        candidates = []

        for nearby_item in predictions[index + 1 : index + 4]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                candidates.append(amount)

        if candidates:
            vatable_sales = extract_vatable_sales(predictions)

            if vatable_sales is not None:
                expected_vat = vatable_sales * 0.12

                return min(
                    candidates,
                    key=lambda amount: abs(amount - expected_vat),
                )

            return candidates[0]

        # Fall back to amounts before the label.
        for nearby_item in predictions[max(0, index - 2) : index]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

    return None


def extract_total(predictions):
    """Extract the receipt total using explicit labels and nearby amounts."""

    total_labels = [
        r"^total$",
        r"^total\s+sales$",
        r"^amount\s+due:?$",
        r"^total\s+payment$",
        r"^grand\s+total$",
    ]

    # Prefer amounts associated with explicit total labels.
    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not any(
            re.fullmatch(pattern, text, re.IGNORECASE) for pattern in total_labels
        ):
            continue

        # Prefer amounts appearing after the total label.
        for nearby_item in predictions[index + 1 : index + 4]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

    # Use LayoutLM B-TOTAL only as a fallback.
    for item in predictions:
        if item["label"] != "B-TOTAL":
            continue

        amount = extract_amount(item["text"])

        if amount is not None:
            return amount

    return None


def validate_vat(vatable_sales, vat_amount):
    """Check whether VAT is approximately 12% of Vatable Sales."""
    if vatable_sales is None or vat_amount is None:
        return False

    expected_vat = vatable_sales * 0.12

    return abs(expected_vat - vat_amount) <= 0.02


def validate_total(vatable_sales, vat_amount, total):
    """Check whether Vatable Sales plus VAT matches the receipt total."""
    if vatable_sales is None or vat_amount is None or total is None:
        return False

    expected_total = vatable_sales + vat_amount

    return abs(expected_total - total) <= 0.02


def extract_receipt_fields(predictions):
    """Extract and validate the main receipt fields."""
    vatable_sales = extract_vatable_sales(predictions)
    vat_amount = extract_vat_amount(predictions)
    total = extract_total(predictions)

    return {
        "company": extract_company(predictions),
        "date": extract_date(predictions),
        "tin": extract_tin(predictions),
        "invoice_number": extract_invoice_number(predictions),
        "vatable_sales": vatable_sales,
        "vat_amount": vat_amount,
        "total": total,
        "vat_valid": validate_vat(vatable_sales, vat_amount),
        "total_valid": validate_total(vatable_sales, vat_amount, total),
    }
