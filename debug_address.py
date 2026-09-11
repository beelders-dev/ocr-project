from src.dataset_generator import find_address_words
from src.dataset_generator import normalize_text, find_exact_sequence

TESTS = {
    "149": "19 & 19A, JALAN MERANTI 2A, SEKSYEI BANDAR BARU BATANG KALI, 44300",
    "152": "USJ SUMMIT, SUBANG (SS)",
    "167": "HQ: 7, JLN SS21/34, 47400 PJ",
    "189": "9, JALAN SUBANG JASA 3, 40150 SHAN ALAM, SELANGOR.",
    "202": "DAISO JAPAN, IOI MALL",
    "236": "NO.12, JALAN SS4C/5,PETALING JAYA SELANGOR DARUL EHSAN",
    "238": "69 JALAN NOVA U5/N, SEKSYEN U5 SUBANG BE",
    "239": "PT17920 SEK U9, SHAH ALAM",
    "264": "PT17920 SEK U9, SHAH ALAM",
    "414": "3, JALAN PERDANA 5, TAMAN INDAH PERDANA, KEPONG, 52100 KL.",
    "435": "1, JALAN SUTERA TANJUNG 8/4, TMN SUTERA UTAMA, 81300 SKUDAI, JB.",
    "463": "NO,35, JALAN KEBUDAYAAN 8, TAMAN UNIVERSITY,81300 SKUDAI, JOHOR",
}


import json

with open("data/processed_dataset.json") as f:
    dataset = json.load(f)


for receipt in dataset:
    receipt_id = receipt["id"]

    if receipt_id not in TESTS:
        continue

    indexes = find_address_words(
        receipt["words"],
        receipt["boxes"],
        TESTS[receipt_id],
    )
    print("OCR WORDS:", receipt["words"])

    print(f"\nID: {receipt_id}")
    print("GT:", TESTS[receipt_id])
    print("MATCH:", indexes)

    if indexes:
        print(
            "MATCHED WORDS:",
            [receipt["words"][i] for i in indexes],
        )


receipt = next(r for r in dataset if r["id"] == "238")

print("OCR:")
for i, word in enumerate(receipt["words"]):
    print(i, repr(word), "=>", normalize_text(word))

print("\nGT:", TESTS["238"])
print(
    "EXACT SEQUENCE:",
    find_exact_sequence(
        receipt["words"],
        TESTS["238"],
    ),
)
