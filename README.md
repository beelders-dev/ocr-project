# Receipt OCR & Information Extraction

An OCR-based receipt information extraction system that uses **PaddleOCR** and **LayoutLMv3** to identify key fields from scanned receipts.

## Project Overview

The project processes receipt images and extracts structured information such as:

- Company
- Date
- Address
- Total

The system combines OCR, document layout information, and visual information to improve field extraction from receipts with different layouts.

## Pipeline

Receipt Image
↓
PaddleOCR
↓
Text + Bounding Boxes
↓
Dataset Preparation
↓
LayoutLMv3
↓
Token Classification
↓
Extracted Receipt Fields

## Tech Stack

- Python 3.12
- PaddleOCR
- PaddlePaddle
- PyTorch
- Hugging Face Transformers
- LayoutLMv3
- scikit-learn
- Pillow

## Project Structure

ocr_project/
├── src/
│ ├── dataset.py
│ ├── dataset_generator.py
│ ├── train.py
│ ├── validate_dataset.py
│ ├── ocr_engine.py
│ ├── extraction/
│ ├── ocr/
│ └── parser/
│
├── data/ # Local datasets and receipt images
├── models/ # Local trained model checkpoints
├── output/ # Generated outputs
├── checkpoints/ # Training checkpoints
│
├── .gitignore
└── README.md

## Dataset

The project uses the **SROIE (Scanned Receipts OCR and Information Extraction)** dataset for training and evaluation.

The current dataset contains:

- 626 receipts
- 500 training receipts
- 126 validation receipts

The dataset is generated locally and is excluded from Git.

## Labels

The LayoutLMv3 model currently performs token classification using:

O
B-COMPANY
I-COMPANY
B-DATE
I-DATE
B-ADDRESS
I-ADDRESS
B-TOTAL
I-TOTAL

`B-` indicates the beginning of a field and `I-` indicates a continuation of the same field.

## Setup

Create and activate a Python 3.12 virtual environment:

python -m venv .venv

Windows PowerShell:

.venv\Scripts\Activate.ps1

Install the required dependencies:

pip install -r requirements.txt

## Generate the Dataset

From the project root:

python -m src.dataset_generator

This generates the processed dataset locally under:

data/processed_dataset.json

## Validate the Dataset

Run:

python -m src.validate_dataset

This performs basic checks on the generated field labels, including potential issues with TOTAL labels.

## Train LayoutLMv3

Run:

python -m src.train

The trained model is saved locally under:

models/

Model files are excluded from Git because of their size.

## Current Status

The project currently has a working prototype for:

- Receipt OCR using PaddleOCR
- OCR word and bounding-box extraction
- SROIE dataset preparation
- Entity label generation
- Bounding-box normalization
- LayoutLMv3 training
- Token-level evaluation

The next stage is to complete and test the **end-to-end inference pipeline**:

Receipt Image → PaddleOCR → LayoutLMv3 → Extracted Fields

## Notes

This repository is the newer implementation of the receipt OCR project and uses a reorganized project structure focused on OCR, dataset preparation, and LayoutLMv3-based information extraction.
