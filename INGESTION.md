# Turkish Legal RAG - Data Ingestion Pipeline

This document explains how to use the data ingestion pipeline to load and normalize Turkish legal datasets.

## Overview

The ingestion pipeline handles:
- Loading datasets from multiple sources (HuggingFace, Kaggle, CSV, JSON, JSONL)
- Careful Turkish text normalization while preserving Turkish characters (ç, ğ, ı, ö, ş, ü)
- Converting different dataset formats to unified QA schema
- Removing duplicates and invalid records
- Comprehensive logging and error handling

## Unified Schema

### QA Pair Schema

All QA datasets are converted to this unified format:

```json
{
  "id": "unique_identifier",
  "question": "Question text in Turkish",
  "answer": "Answer text in Turkish",
  "source": "source_name",
  "category": "Legal category (optional)",
  "citation": "Reference or citation (optional)"
}
```

### Legal Document Schema

Legal documents are converted to:

```json
{
  "doc_id": "unique_document_id",
  "title": "Document title",
  "law_name": "Name of law/statute (optional)",
  "article_no": "Article number (optional)",
  "section": "Section or subsection (optional)",
  "text": "Full legal text",
  "source": "source_name"
}
```

## Usage Examples

### 1. Demo with Example Data

To test the pipeline with provided example files:

```bash
python scripts/ingest_demo.py
```

This will:
- Process example lawchatbot dataset
- Process example Kaggle-style dataset
- Save outputs to `data/processed/`
- Display normalization examples

Output files:
- `example_lawchatbot_processed.jsonl`
- `example_kaggle_turkishlaw_processed.jsonl`

### 2. Load HuggingFace Dataset Only

To load the Renicames/turkish-lawchatbot dataset from HuggingFace:

```bash
python scripts/ingest_turkish_datasets.py --hf
```

Requirements:
- `datasets` library installed: `pip install datasets`
- Internet connection to download from HuggingFace Hub

Output:
- `data/processed/huggingface_turkish_lawchatbot.jsonl` - Full dataset
- `data/processed/huggingface_turkish_lawchatbot_preview.json` - First 10 samples

### 3. Load Kaggle Dataset Only

First, download the dataset from Kaggle and extract to `data/raw/kaggle_turkishlaw/`:

```bash
python scripts/ingest_turkish_datasets.py --kaggle data/raw/kaggle_turkishlaw
```

The script will:
- Scan the folder for CSV, JSON, and JSONL files
- Auto-detect column names (question/answer/instruction/output patterns)
- Convert to unified schema

Output:
- `data/processed/kaggle_turkishlaw.jsonl` - Full dataset
- `data/processed/kaggle_turkishlaw_preview.json` - First 10 samples

### 4. Load Both Datasets

To load and combine both datasets:

```bash
python scripts/ingest_turkish_datasets.py --all data/raw/kaggle_turkishlaw
```

This will:
- Load HuggingFace dataset
- Load Kaggle dataset
- Combine into single file
- Report duplicate statistics

Output:
- `data/processed/combined_turkish_legal_qa.jsonl` - Combined dataset with stats

## Python API

You can also use the ingestion modules directly in your code:

### Loading Data

```python
from src.ingestion.loader import load_data, load_huggingface_dataset, load_kaggle_dataset_from_folder

# Load from file (auto-detects format: CSV, JSON, JSONL)
data = load_data('data/raw/my_dataset.json')

# Load from HuggingFace
hf_data = load_huggingface_dataset('Renicames/turkish-lawchatbot', split='train')

# Load from Kaggle folder
kaggle_data = load_kaggle_dataset_from_folder('data/raw/kaggle_turkishlaw')
```

### Converting Datasets

```python
from src.ingestion.converters import TurkishLawchatbotConverter, TurkishLawKaggleConverter, QAConverter

# For HuggingFace lawchatbot dataset
converter = TurkishLawchatbotConverter()
qa_pairs = converter.convert_batch(raw_data)

# For Kaggle turkishlaw dataset
converter = TurkishLawKaggleConverter()
qa_pairs = converter.convert_batch(raw_data)

# For custom datasets with known column names
converter = QAConverter(field_mapping={
    'question': 'q',
    'answer': 'a',
    'category': 'topic'
})
qa_pairs = converter.convert_batch(raw_data, source_name='my_source')
```

### Text Normalization

```python
from src.ingestion.normalizer import normalize_turkish_text

# Normalize Turkish text
text = "  Türkiye'de   hukuk    sistemi  nasıl  çalışır?  "
normalized = normalize_turkish_text(text)
# Result: "Türkiye'de hukuk sistemi nasıl çalışır?"

# Keep original case
normalized = normalize_turkish_text(text, preserve_case=True)
```

## Supported Datasets

### 1. HuggingFace: Renicames/turkish-lawchatbot

- **Source**: https://huggingface.co/datasets/Renicames/turkish-lawchatbot
- **Format**: QA pairs
- **Fields**: question, answer, category
- **Size**: ~1000 QA pairs
- **License**: Check HuggingFace repo

### 2. Kaggle: batuhankalem/turkishlaw-dataset-for-llm-finetuning

- **Source**: https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning
- **Format**: Instruction-following format
- **Fields**: instruction, input, output (auto-detected)
- **Size**: Variable
- **License**: Check Kaggle repo

## Output Formats

All outputs are saved as **JSONL** (JSON Lines):
- One JSON object per line
- Easy to stream and process
- Compatible with Hugging Face datasets library

Example:
```jsonl
{"id": "id1", "question": "...", "answer": "...", "source": "...", ...}
{"id": "id2", "question": "...", "answer": "...", "source": "...", ...}
```

## Turkish Character Handling

The pipeline carefully preserves Turkish characters:

- **Lowercase**: ç, ğ, ı, ö, ş, ü
- **Uppercase**: Ç, Ğ, İ, Ö, Ş, Ü

Normalization steps:
1. Remove control characters (preserves Turkish chars)
2. Normalize whitespace (multiple spaces → single space)
3. Handle quote variations (curly quotes → straight quotes)
4. Optional lowercase conversion (Turkish-aware)

## Data Quality

The pipeline performs:

✓ **Text cleaning**: Removes control characters and excess whitespace
✓ **Validation**: Ensures required fields are present
✓ **Deduplication**: Removes exact duplicates by question text (case-insensitive)
✓ **Error handling**: Logs skipped records with reasons
✓ **Logging**: Comprehensive logging of ingestion process

## Troubleshooting

### "Kaggle dataset path does not exist"

Make sure you've downloaded the Kaggle dataset and extracted it to the correct path:
```bash
mkdir -p data/raw/kaggle_turkishlaw
# Extract Kaggle files here
```

### "datasets library not installed"

Install the datasets library:
```bash
pip install datasets
```

### No output files created

Check the log messages for errors. Common issues:
- Dataset format doesn't match expected schema
- Missing required columns
- Character encoding issues (ensure UTF-8)

### Column names not auto-detected for Kaggle dataset

The pipeline looks for common column patterns:
- Questions: question, instruction, prompt, soru, q
- Answers: answer, output, response, cevap, a

If your dataset uses different names, provide custom mapping:
```python
converter = QAConverter(field_mapping={
    'question': 'your_question_column',
    'answer': 'your_answer_column'
})
```

## Next Steps

After ingestion, your processed data is ready for:

1. **Retrieval**: Build embeddings and create vector indices in `src/retrieval/`
2. **Reranking**: Fine-tune rerankers in `src/reranking/`
3. **Fine-tuning**: Fine-tune LLMs in `src/llm/`
4. **Evaluation**: Evaluate system performance in `src/evaluation/`

See respective module documentation for details.

## Performance Notes

- HuggingFace dataset: ~10MB, ~1000 records
- Kaggle dataset: Variable, typically 5-20MB
- Processing time: <1 second per dataset
- Memory usage: ~500MB for both datasets

## Contributing

To add support for new datasets:

1. Create a new converter class in `src/ingestion/converters.py`
2. Inherit from `QAConverter` or `LegalTextConverter`
3. Override `convert_batch()` with custom logic
4. Update the main ingestion script with new option
5. Add documentation here

Example:

```python
class MyDatasetConverter(QAConverter):
    def convert_batch(self, records):
        # Custom conversion logic
        return converted_records
```

## License

See main project README for license information.
