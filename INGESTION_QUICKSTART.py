#!/usr/bin/env python3
"""
Quick reference guide for Turkish Legal RAG Data Ingestion

This file shows common operations with minimal examples.
"""

# ============================================================================
# QUICK START: Data Ingestion
# ============================================================================

# 1. DEMO WITH EXAMPLE DATA
# --------------------------
# Best for getting started quickly without downloading datasets
# 
# $ python scripts/ingest_demo.py
#
# This processes example Turkish legal QA files already in data/raw/
# Output: data/processed/example_*.jsonl


# 2. LOAD HUGGINGFACE DATASET
# ----------------------------
# Requires: pip install datasets
#
# $ python scripts/ingest_turkish_datasets.py --hf
#
# Downloads: Renicames/turkish-lawchatbot from HuggingFace Hub
# Output: data/processed/huggingface_turkish_lawchatbot.jsonl


# 3. LOAD KAGGLE DATASET
# ----------------------
# Steps:
# a) Download from https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning
# b) Extract to: data/raw/kaggle_turkishlaw/
# c) Run:
#
# $ python scripts/ingest_turkish_datasets.py --kaggle data/raw/kaggle_turkishlaw
#
# Output: data/processed/kaggle_turkishlaw.jsonl


# 4. COMBINE BOTH DATASETS
# -------------------------
# $ python scripts/ingest_turkish_datasets.py --all data/raw/kaggle_turkishlaw
#
# Output: data/processed/combined_turkish_legal_qa.jsonl (with duplicate report)


# ============================================================================
# IN-CODE EXAMPLES
# ============================================================================

# Example 1: Load and normalize text
# -----------------------------------
#
# from src.ingestion.normalizer import normalize_turkish_text
#
# text = "  Türkiye'de  hukuk   sistemi nasıl çalışır?  "
# clean_text = normalize_turkish_text(text)
# print(clean_text)  # Output: "Türkiye'de hukuk sistemi nasıl çalışır?"


# Example 2: Load dataset from file
# ----------------------------------
#
# from src.ingestion.loader import load_data, inspect_dataset_schema
#
# raw_data = load_data('data/raw/my_dataset.json')
# inspect_dataset_schema(raw_data, num_samples=2)


# Example 3: Convert custom dataset
# ----------------------------------
#
# from src.ingestion.converters import QAConverter
#
# converter = QAConverter(field_mapping={
#     'question': 'q',
#     'answer': 'a',
#     'category': 'topic'
# })
# qa_pairs = converter.convert_batch(raw_data, source_name='my_source')
#
# # Save output
# import json
# with open('output.jsonl', 'w') as f:
#     for record in qa_pairs:
#         f.write(json.dumps(record, ensure_ascii=False) + '\n')


# Example 4: Load from HuggingFace programmatically
# --------------------------------------------------
#
# from src.ingestion.loader import load_huggingface_dataset
# from src.ingestion.converters import TurkishLawchatbotConverter
#
# raw_data = load_huggingface_dataset('Renicames/turkish-lawchatbot', split='train')
# converter = TurkishLawchatbotConverter()
# qa_pairs = converter.convert_batch(raw_data)


# Example 5: Load from Kaggle folder
# -----------------------------------
#
# from src.ingestion.loader import load_kaggle_dataset_from_folder
# from src.ingestion.converters import TurkishLawKaggleConverter
#
# raw_data = load_kaggle_dataset_from_folder('data/raw/kaggle_turkishlaw')
# converter = TurkishLawKaggleConverter()
# qa_pairs = converter.convert_batch(raw_data)


# ============================================================================
# OUTPUT FORMAT
# ============================================================================
#
# All processed data is saved as JSONL (one JSON object per line):
#
# {"id": "abc123", "question": "Soru metni", "answer": "Cevap metni", ...}
# {"id": "def456", "question": "Başka soru", "answer": "Başka cevap", ...}
#
# Fields:
#   - id: Unique identifier
#   - question: Question in Turkish
#   - answer: Answer in Turkish
#   - source: Where data came from (huggingface, kaggle, etc)
#   - category: Legal category (optional)
#   - citation: Reference (optional)


# ============================================================================
# FILE STRUCTURE
# ============================================================================
#
# src/ingestion/
#   ├── normalizer.py       - Text normalization functions
#   ├── loader.py           - Data loading (CSV, JSON, HuggingFace, Kaggle)
#   ├── converters.py       - Schema converters (QA, Legal Text)
#   ├── schema.py           - Data schema definitions
#   └── pipeline.py         - Full ingestion pipeline orchestrator
#
# scripts/
#   ├── ingest_demo.py              - Quick demo with example data
#   └── ingest_turkish_datasets.py  - Load real Turkish datasets
#
# data/
#   ├── raw/
#   │   ├── example_lawchatbot.json           - Example LawChatBot data
#   │   ├── example_kaggle_turkishlaw.json    - Example Kaggle data
#   │   └── kaggle_turkishlaw/                - Kaggle dataset folder (manual download)
#   └── processed/
#       ├── example_lawchatbot_processed.jsonl
#       ├── example_kaggle_turkishlaw_processed.jsonl
#       ├── huggingface_turkish_lawchatbot.jsonl
#       ├── kaggle_turkishlaw.jsonl
#       └── combined_turkish_legal_qa.jsonl


# ============================================================================
# COMMON ISSUES & SOLUTIONS
# ============================================================================
#
# Q: "datasets library not installed"
# A: Run: pip install datasets
#
# Q: "Kaggle dataset path does not exist"
# A: Make sure you downloaded the Kaggle dataset and extracted it to:
#    data/raw/kaggle_turkishlaw/
#
# Q: "No Turkish characters in output"
# A: Ensure files are UTF-8 encoded. The pipeline preserves Turkish chars
#    (ç, ğ, ı, ö, ş, ü, etc.)
#
# Q: "Columns not auto-detected for Kaggle"
# A: The pipeline looks for common patterns. For custom datasets, use:
#    QAConverter(field_mapping={'question': 'your_col_name', ...})


# ============================================================================
# WHAT TO DO NEXT
# ============================================================================
#
# After ingestion, your cleaned data is ready for:
#
# 1. RETRIEVAL (src/retrieval/)
#    - Build embeddings with SentenceTransformers
#    - Index with FAISS
#    - Enable semantic search
#
# 2. RERANKING (src/reranking/)
#    - Fine-tune cross-encoders
#    - Implement BM25 ranking
#    - Improve result quality
#
# 3. LLM FINE-TUNING (src/llm/)
#    - Fine-tune Turkish LLMs with PEFT
#    - Use data for instruction tuning
#    - Improve legal domain knowledge
#
# 4. EVALUATION (src/evaluation/)
#    - Measure retrieval metrics
#    - Evaluate generation quality
#    - Track performance over time


print("""
╔════════════════════════════════════════════════════════════════════════════╗
║       Turkish Legal RAG - Data Ingestion Quick Reference Guide            ║
║                                                                             ║
║  Quick Start:                                                              ║
║    1. Demo with examples:  python scripts/ingest_demo.py                  ║
║    2. Load HuggingFace:    python scripts/ingest_turkish_datasets.py --hf ║
║    3. Load Kaggle:         python scripts/ingest_turkish_datasets.py \\     ║
║                            --kaggle data/raw/kaggle_turkishlaw            ║
║                                                                             ║
║  See INGESTION.md for detailed documentation                              ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
