import json
import re

DATASET_PATH = "data/processed_dataset.json"

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

TOTAL_LABELS = {"B-TOTAL", "I-TOTAL"}

STRONG_TOTAL_KEYWORDS = [
    "TOTAL",
    "TOTAL SALES",
    "TOTAL AMOUNT",
    "TOTAL AMT",
    "TOTAL DUE",
    "TOTAL PAYABLE",
    "AMOUNT TO BE PAID",
    "GRAND TOTAL",
]

BAD_TOTAL_CONTEXT = [
    "SUBTOTAL",
    "SUB TOTAL",
    "CASH",
    "CASH RECEIVED",
    "CASH TENDERED",
    "CHANGE",
    "DISCOUNT",
]


def normalize(text):
    return re.sub(r"\s+", " ", str(text).upper()).strip()


def is_amount(text):
    text = str(text).replace(",", "").strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d{1,2})?", text))


def get_label_name(label):
    if isinstance(label, int):
        return LABEL_LIST[label]

    return str(label)


def find_nearby_label(words, total_index):
    """
    Look backward from the TOTAL value and identify the nearest
    meaningful label.
    """
    for i in range(total_index - 1, max(-1, total_index - 6), -1):
        word = normalize(words[i])

        if not word:
            continue

        for keyword in STRONG_TOTAL_KEYWORDS:
            if keyword in word:
                return "TOTAL", keyword

        for keyword in BAD_TOTAL_CONTEXT:
            if keyword in word:
                return "BAD", keyword

    return None, None


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    good = 0
    duplicate = 0
    suspicious = []

    for receipt_index, sample in enumerate(dataset):
        words = sample.get("words", [])
        labels = sample.get("ner_tags", [])

        label_names = [get_label_name(label) for label in labels]

        total_indices = [
            i for i, label in enumerate(label_names) if label in TOTAL_LABELS
        ]

        if not total_indices:
            continue

        total_index = total_indices[0]
        total_word = words[total_index]

        nearby_type, nearby_keyword = find_nearby_label(
            words,
            total_index,
        )

        reasons = []

        # A TOTAL directly associated with a valid total label is good.
        if nearby_type == "TOTAL":
            good += 1

        # Flag only when a bad label is closer than a valid TOTAL label.
        elif nearby_type == "BAD":
            reasons.append(f"near {nearby_keyword}")
            suspicious.append(
                {
                    "receipt": receipt_index,
                    "total": total_word,
                    "reasons": reasons,
                    "context": words[max(0, total_index - 6) : total_index + 1],
                }
            )

        # If no useful label was found, check for duplicates.
        else:
            if is_amount(total_word):
                duplicate_indices = [
                    i
                    for i, word in enumerate(words)
                    if i != total_index and normalize(word) == normalize(total_word)
                ]

                if duplicate_indices:
                    duplicate += 1

    print("=" * 70)
    print("TOTAL LABEL VALIDATION")
    print("=" * 70)

    print(f"\nReceipts: {len(dataset)}")
    print(f"Good TOTAL labels: {good}")
    print(f"Duplicate amounts: {duplicate}")
    print(f"Suspicious TOTAL labels: {len(suspicious)}")

    if not suspicious:
        print("\nNo suspicious TOTAL labels found.")
        print("\nDataset is ready for retraining.")
        return

    print("\nSuspicious cases:")
    print("-" * 70)

    for item in suspicious:
        print(f"\nReceipt {item['receipt']}")
        print(f"TOTAL: {item['total']}")
        print(f"Reason: {', '.join(item['reasons'])}")

        print("Context:")
        for word in item["context"]:
            print(f"  {word}")

    print("\n" + "=" * 70)
    print("Review these receipts before retraining.")
    print("=" * 70)


if __name__ == "__main__":
    main()
