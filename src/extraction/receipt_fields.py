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
    pattern = r"\b\d{2}[-/]\d{2}[-/]\d{4}(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b"

    candidates = []

    for item in predictions:
        match = re.search(pattern, item["text"])

        if match:
            candidates.append(match.group())

    # Prefer a date that includes a time.
    for candidate in candidates:
        if re.search(r"\d{2}:\d{2}", candidate):
            return candidate

    return candidates[-1] if candidates else ""


def extract_tin(predictions):
    """Extract a Philippine TIN from receipt text."""
    for item in predictions:
        match = re.search(TIN_PATTERN, item["text"])

        if match and "TIN" in item["text"].upper():
            return match.group()

    # Also check nearby OCR words.
    for index, item in enumerate(predictions):
        if "TIN" in item["text"].upper():
            nearby_text = " ".join(
                next_item["text"] for next_item in predictions[index : index + 3]
            )

            match = re.search(TIN_PATTERN, nearby_text)

            if match:
                return match.group()

    return ""


def extract_invoice_number(predictions):
    """Extract the sales invoice number from its nearby label."""
    for index, item in enumerate(predictions):
        text = item["text"].upper()

        if "SALES INVOICE" in text and "NO" in text:
            match = re.search(r"\b\d{4,}\b", item["text"])

            if match:
                return match.group()

            nearby_text = " ".join(
                next_item["text"] for next_item in predictions[index : index + 3]
            )

            match = re.search(r"\b\d{4,}\b", nearby_text)

            if match:
                return match.group()

    return ""


def extract_labeled_amount(predictions, label_pattern):
    """Find the numeric value associated with a nearby receipt label."""
    for index, item in enumerate(predictions):
        if re.search(label_pattern, item["text"], re.IGNORECASE):
            for next_item in predictions[index + 1 : index + 3]:
                match = re.search(AMOUNT_PATTERN, next_item["text"])

                if match:
                    return float(match.group().replace(",", ""))

    return None


def extract_vatable_sales(predictions):
    """Extract the amount associated with Vatable Sale."""
    return extract_labeled_amount(
        predictions,
        r"[V/]\s*atable\s+Sale",
    )


def extract_vat_amount(predictions):
    """Extract the amount associated with VAT (12%)."""
    return extract_labeled_amount(
        predictions,
        r"[V/]\s*AT\s*\(?12%\)?",
    )


def extract_total(predictions):
    """Extract the amount immediately following TOTAL."""
    for index, item in enumerate(predictions):
        if re.fullmatch(
            r"TOTAL",
            item["text"].strip(),
            re.IGNORECASE,
        ):
            for next_item in predictions[index + 1 : index + 3]:
                match = re.search(
                    AMOUNT_PATTERN,
                    next_item["text"],
                )

                if match:
                    return float(match.group().replace(",", ""))

    return None


def validate_vat(vatable_sales, vat_amount):
    """Check whether VAT is approximately 12% of vatable sales."""
    if vatable_sales is None or vat_amount is None:
        return False

    expected_vat = vatable_sales * 0.12

    return abs(expected_vat - vat_amount) <= 0.02


def extract_receipt_fields(predictions):
    """Extract the main receipt fields from OCR and LayoutLM predictions."""
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
