import re

AMOUNT_PATTERN = r"(?<!\d)\d{1,3}(?:,\d{3})*\.\d{2}(?!\d)"
TIN_PATTERN = r"\b\d{3}[-\s]\d{3}[-\s]\d{3}[-\s]\d{3,5}\b"


def extract_company(predictions):
    """Extract the company name while removing overlapping predictions."""
    company_items = [
        item["text"].strip()
        for item in predictions
        if item["label"] in {"B-COMPANY", "I-COMPANY"} and item["text"].strip()
    ]

    if not company_items:
        return ""

    # Prefer a complete prediction when one contains the other
    # company predictions.
    for candidate in sorted(company_items, key=len, reverse=True):
        normalized_candidate = " ".join(candidate.upper().split())

        contained = True

        for other in company_items:
            normalized_other = " ".join(other.upper().split())

            if normalized_other == normalized_candidate:
                continue

            if normalized_other not in normalized_candidate:
                contained = False
                break

        if contained:
            return candidate

    return " ".join(company_items)


def extract_date(predictions):
    """Extract the most likely transaction date from receipt text."""
    date_patterns = [
        r"\b\d{2}[-/]\d{2}[-/]\d{4}",
        r"\b\d{2}[-/]\d{2}[-/]\d{2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}\b",
    ]

    excluded_context = [
        "DATEISSUED",
        "DATEI88UED",
        "DATEOFISSUANCE",
        "EXPIR",
        "VALIDUNTIL",
        "DUEDATE",
    ]

    candidates = []

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not text:
            continue

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)

            if not match:
                continue

            nearby_text = " ".join(
                predictions[i]["text"].upper()
                for i in range(
                    max(0, index - 1),
                    min(len(predictions), index + 2),
                )
            )

            # Normalize spacing so variants like
            # "DateIssued" and "Date Issued" match the same rule.
            normalized_context = "".join(nearby_text.split())

            if any(context in normalized_context for context in excluded_context):
                continue

            candidates.append(
                {
                    "value": match.group(),
                    "index": index,
                }
            )

            break

    if not candidates:
        return ""

    # Prefer the latest remaining date in the receipt.
    return candidates[-1]["value"]


def extract_tin(predictions):
    """Extract the merchant TIN using receipt position and nearby context."""
    tin_candidates = []

    for index, item in enumerate(predictions):
        text = item["text"].strip()
        upper = text.upper()

        # OCR may read TIN as T1N, T!N, etc.
        if not re.search(r"T[1I!]N", upper):
            continue

        match = re.search(TIN_PATTERN, text)

        if not match:
            nearby_text = " ".join(
                next_item["text"]
                for next_item in predictions[max(0, index - 2) : index + 3]
            )

            match = re.search(TIN_PATTERN, nearby_text)

        if not match:
            continue

        value = match.group()

        score = 0

        # Merchant TINs are usually near the top/header.
        if index < 20:
            score += 10

        # "VAT Reg." near the TIN is a strong merchant indicator.
        nearby = " ".join(
            item["text"].upper() for item in predictions[max(0, index - 2) : index + 2]
        )

        if "VAT REG" in nearby:
            score += 10

        # Footer/vendor information should be strongly penalized.
        if index > len(predictions) * 0.7:
            score -= 15

        if any(
            keyword in nearby
            for keyword in [
                "SOFTWARE",
                "FCCDTN",
                "DATE OF ISSUANCE",
                "ACCREDIT",
            ]
        ):
            score -= 15

        tin_candidates.append(
            {
                "value": value,
                "score": score,
                "index": index,
            }
        )

    if not tin_candidates:
        return ""

    tin_candidates.sort(
        key=lambda candidate: (
            candidate["score"],
            -candidate["index"],
        ),
        reverse=True,
    )

    return tin_candidates[0]["value"]


def extract_invoice_number(predictions):
    """Extract an invoice number from an invoice label and nearby OCR text."""

    invoice_label_pattern = re.compile(
        r"(?:SALES\s+INVOICE|INVOICE|INV0ICE|INVO1CE|INV|SI)"
        r"(?:\s*(?:#|№|NO\.?|NUMBER))?",
        re.IGNORECASE,
    )

    number_pattern = re.compile(r"(?<!\d)\d{6,}(?:-\d+)?(?!\d)")

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not invoice_label_pattern.search(text):
            continue

        # First check the label itself.
        match = number_pattern.search(text)

        if match:
            return match.group()

        # Only inspect the next two OCR items.
        nearby_items = predictions[index + 1 : index + 3]

        for nearby_item in nearby_items:
            candidate_text = nearby_item["text"]

            # Stop if we have already moved into unrelated receipt sections.
            if re.search(
                r"ACCREDIT|DATE\s*ISSUED|DATE\s+OF\s+ISSUANCE|VALID\s+UNTIL",
                candidate_text,
                re.IGNORECASE,
            ):
                break

            match = number_pattern.search(candidate_text)

            if match:
                return match.group()

    return ""


def extract_amount(text):
    """Extract a two-decimal receipt amount from OCR text."""
    match = re.search(AMOUNT_PATTERN, text)

    if not match:
        return None

    return float(match.group().replace(",", ""))


def extract_labeled_amount(predictions, label_pattern):
    """Find an amount associated with a nearby receipt label."""
    for index, item in enumerate(predictions):
        if not re.search(
            label_pattern,
            item["text"],
            re.IGNORECASE,
        ):
            continue

        nearby_items = predictions[index + 1 : index + 4]

        for nearby_item in nearby_items:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

    return None


def extract_vatable_sales(predictions):
    """Extract Vatable Sales using receipt labels and nearby VAT amounts."""

    vatable_patterns = [
        r"^/?[vu]?atable(?:\s+sales?)?:?$",
        r"^vat\s+sales:?$",
        r"^sales\s+excl\.?\s*(?:of\s+)?vat:?$",
        r"^net\s+sales:?$",
    ]

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not any(
            re.fullmatch(pattern, text, re.IGNORECASE) for pattern in vatable_patterns
        ):
            continue

        # Collect nearby numeric values on both sides of the label.
        nearby_items = predictions[max(0, index - 3) : index + 4]

        amounts = []

        for nearby_item in nearby_items:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                amounts.append(amount)

        # Prefer a pair that follows the expected 12% VAT relationship.
        vatable_sales, vat_amount = find_vat_pair(amounts)

        if vatable_sales is not None:
            return vatable_sales

        # Preserve the previous behavior as a fallback.
        for nearby_item in predictions[index + 1 : index + 4]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

        for nearby_item in predictions[max(0, index - 3) : index]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

    return None


def extract_vat_amount(predictions):
    """Extract VAT Amount using labels and the expected 12% relationship."""
    vat_patterns = [
        r"^/?vat:?$",
        r"^/?vat\s+amount:?$",
        r"^/?uat\s+amount:?$",
        r"^/?vat\s*[-:]?\s*(?:[\(\（]?\s*12[%％号]?\s*[\)\）]?)?:?$",
        r"^12[%％号]?\s+vat:?$",
        r"^plus\s+vat\s+amount:?$",
        r"^vat\s*[-:]?\s*12[%％号]?:?$",
    ]

    vatable_sales = extract_vatable_sales(predictions)

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not any(
            re.fullmatch(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in vat_patterns
        ):
            continue

        candidates = []

        for nearby_item in predictions[max(0, index - 3) : index + 4]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                candidates.append(amount)

        if not candidates:
            continue

        if vatable_sales is not None:
            expected_vat = vatable_sales * 0.12

            return min(
                candidates,
                key=lambda amount: abs(amount - expected_vat),
            )

        return candidates[0]

    return None


def extract_total(predictions):
    """Extract the receipt total using explicit labels and nearby amounts."""
    total_labels = [
        r"^total$",
        r"^total\s+sales$",
        r"^total\s+salas$",
        r"^total\s+sale?s$",
        r"^amount\s+due:?$",
        r"^total\s+payment$",
        r"^grand\s+total$",
    ]

    for index, item in enumerate(predictions):
        text = item["text"].strip()

        if not any(
            re.fullmatch(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in total_labels
        ):
            continue

        for nearby_item in predictions[index + 1 : index + 4]:
            amount = extract_amount(nearby_item["text"])

            if amount is not None:
                return amount

    # LayoutLM total prediction remains a fallback.
    for item in predictions:
        if item["label"] != "B-TOTAL":
            continue

        amount = extract_amount(item["text"])

        if amount is not None:
            return amount

    # Mathematical fallback.
    vatable_sales = extract_vatable_sales(predictions)
    vat_amount = extract_vat_amount(predictions)

    if vatable_sales is not None and vat_amount is not None:
        return round(vatable_sales + vat_amount, 2)

    return None


def find_vat_pair(amounts):
    """Find a Vatable Sales and VAT Amount pair using the 12% VAT relationship."""
    for first in amounts:
        for second in amounts:
            if first == second:
                continue

            base = max(first, second)
            vat = min(first, second)

            if round(base * 0.12, 2) == round(vat, 2):
                return base, vat

    return None, None


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
        "vat_valid": validate_vat(
            vatable_sales,
            vat_amount,
        ),
        "total_valid": validate_total(
            vatable_sales,
            vat_amount,
            total,
        ),
    }
