# Turkish Legal RAG 🏛️

**A Retrieval-Augmented Generation (RAG) system for Turkish legal question answering.**

This is a student research project for implementing a comprehensive RAG pipeline to answer Turkish legal questions by retrieving relevant documents and generating grounded, cited answers.

---

## 📋 Project Overview

### Objective
Build an end-to-end RAG system that:
1. **Retrieves** relevant Turkish legal documents using dense and sparse methods
2. **Reranks** results with cross-encoder models
3. **Generates** grounded answers from context
4. **Evaluates** with both retrieval and QA metrics

### Technology Stack
```
Core: Python 3.8+
NLP: transformers, sentence-transformers
Embeddings: FAISS, sentence-transformers
Ranking: rank-bm25, cross-encoder models
LLM: peft (LoRA/QLoRA for fine-tuning)
Evaluation: scikit-learn, rouge-score
UI: Gradio
```

---

## 📁 Project Structure

```
turkish-legal-rag/
├── data/
│   ├── raw/                    # Original datasets (CSV, JSON, JSONL)
│   ├── processed/              # Cleaned QA pairs & documents (JSONL)
│   ├── chunked/                # Text chunks for retrieval
│   └── gold/                   # Gold evaluation sets (optional)
│
├── src/                        # Main source code
│   ├── ingestion/              # Data loading & normalization
│   │   ├── loader.py           # CSV/JSON/JSONL loading
│   │   ├── normalizer.py       # Turkish text preprocessing
│   │   ├── schema.py           # Data schema definitions
│   │   ├── converters.py       # Format converters
│   │   ├── pipeline.py         # Orchestration
│   │   └── __init__.py
│   ├── retrieval/              # Dense & sparse retrieval
│   │   ├── embeddings.py       # Embedding models
│   │   ├── dense.py            # FAISS-based dense search
│   │   ├── sparse.py           # BM25 sparse search
│   │   ├── hybrid.py           # Hybrid fusion
│   │   └── __init__.py
│   ├── reranking/              # Result reranking
│   │   ├── cross_encoder.py    # Cross-encoder reranker
│   │   ├── eval_reranker.py    # Reranker evaluation
│   │   └── __init__.py
│   ├── llm/                    # LLM generation & fine-tuning
│   │   ├── prompt_builder.py   # Prompt templates
│   │   ├── generation.py       # Answer generation
│   │   ├── finetune.py         # LoRA/QLoRA training
│   │   └── __init__.py
│   ├── evaluation/             # Evaluation metrics
│   │   ├── retrieval_metrics.py # Recall, MRR, nDCG
│   │   ├── qa_metrics.py       # EM, F1, BLEU, ROUGE
│   │   ├── hallucination.py    # Hallucination analysis
│   │   └── __init__.py
│   ├── utils/                  # Utilities
│   │   ├── logging.py          # Logging setup
│   │   ├── config.py           # Config loading
│   │   └── __init__.py
│   └── __init__.py
│
├── scripts/                    # Executable scripts
│   ├── ingest_demo.py          # Demo ingestion
│   ├── ingest_data.py          # Main ingestion
│   ├── ingest_turkish_datasets.py # HF/Kaggle loading
│   ├── chunk_documents.py       # Document chunking
│   ├── build_retrieval_index.py # Build FAISS/BM25
│   ├── train_embeddings.py      # Embedding fine-tuning
│   ├── train_reranker.py        # Cross-encoder training
│   ├── generate_answers.py      # Answer generation
│   ├── run_evaluation.py        # Full evaluation
│   └── run_experiments.py       # Ablation studies
│
├── configs/                    # Configuration files
│   ├── ingestion_config.yaml   # Ingestion settings
│   ├── retrieval_config.yaml   # Retrieval models & params
│   ├── reranking_config.yaml   # Reranker settings
│   ├── generation_config.yaml  # LLM settings
│   └── training_config.yaml    # Training hyperparameters
│
├── notebooks/                  # Jupyter exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_analysis.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_ablation_study.ipynb
│
├── experiments/                # Experiment logs & results
│   ├── baseline/               # Baseline runs
│   ├── with_embedding_tuning/  # Embedding fine-tuning results
│   ├── with_reranking/         # Reranking results
│   ├── with_llm_tuning/        # LLM fine-tuning results
│   └── full_pipeline/          # Full system results
│
├── reports/                    # Reports & analysis
│   ├── metrics_summary.csv     # Aggregated metrics
│   ├── error_analysis.md       # Error breakdown
│   ├── hallucination_report.md # Hallucination findings
│   └── final_report.md         # Technical report
│
├── app/                        # Demo application
│   ├── demo_gradio.py          # Gradio interface
│   └── demo_streamlit.py       # Streamlit interface
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── INGESTION.md                # Data ingestion guide
├── COMPLETE_GUIDE_TR.md        # Turkish setup guide
├── HEALTH_CHECK_REPORT.md      # System status
├── TEST_RESULTS.md             # Test results
├── IMPLEMENTATION_SUMMARY.md   # Implementation notes
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install Python 3.8+ (already done)

# Create virtual environment  
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# OR
source venv/bin/activate         # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import sentence_transformers; print('✅ Setup complete')"
```

### 2. Run Data Ingestion

```bash
# Test with example data
python scripts/ingest_demo.py

# Ingest your own CSV/JSON  
python scripts/ingest_data.py --input data/raw/my_data.json --output my_processed --type qa

# Load HuggingFace datasets
python scripts/ingest_turkish_datasets.py --hf
```

### 3. Chunk Documents

```bash
# Split documents into retrieval chunks
python scripts/chunk_documents.py \
    --input data/processed/my_chunk_input.jsonl \
    --output data/chunked/my_documents_chunks.jsonl \
    --chunk_size 300
```

### 4. Build Retrieval Index

```bash
# Build FAISS + BM25 indices
python scripts/build_retrieval_index.py \
    --documents data/chunked/my_documents_chunks.jsonl \
    --model_name "sentence-transformers/distiluse-base-multilingual-cased-v2" \
    --output models/retrieval_index
```

### 5. Generate Answers

```bash
# Test RAG pipeline end-to-end
python scripts/generate_answers.py \
    --question "Türk ceza kanununda adam öldürme suçu nedir?" \
    --retrieval_index models/retrieval_index \
    --model_name "gpt2"  # or your chosen Turkish LLM
```

### 6. Run Evaluation

```bash
# Evaluate on test set
python scripts/run_evaluation.py \
    --test_set data/gold/test_qa.jsonl \
    --retrieval_index models/retrieval_index \
    --output reports/metrics_summary.csv
```

### 7. Launch Demo App

```bash
# Gradio web interface
python app/demo_gradio.py

# OR Streamlit interface  
streamlit run app/demo_streamlit.py
```

---

## 📊 System Pipeline

```
INPUT QUERY (Turkish legal question)
    ↓
[INGESTION] Load & normalize documents → JSONL
    ↓
[CHUNKING] Split long documents into retrieval units
    ↓
[RETRIEVAL] Dense (FAISS) + Sparse (BM25) search → Top-K candidates
    ↓
[RERANKING] Cross-encoder rescoring → Top-5 relevant chunks
    ↓
[GENERATION] Prompt with context → LLM generates answer
    ↓
OUTPUT: Grounded answer with citations
```

---

## 🎯 Key Components

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **Ingestion** | `src/ingestion/` | Load, normalize, convert data formats |
| **Chunking** | `scripts/chunk_documents.py` | Split documents for retrieval |
| **Dense Retrieval** | `src/retrieval/dense.py` | FAISS vector search |
| **Sparse Retrieval** | `src/retrieval/sparse.py` | BM25 keyword search |
| **Hybrid Fusion** | `src/retrieval/hybrid.py` | Combine dense + sparse |
| **Reranking** | `src/reranking/cross_encoder.py` | Cross-encoder reranker |
| **Prompt Building** | `src/llm/prompt_builder.py` | Context-aware prompts |
| **Generation** | `src/llm/generation.py` | LLM answer generation |
| **Fine-tuning** | `src/llm/finetune.py` | LoRA/QLoRA training |
| **Evaluation** | `src/evaluation/` | Metrics & analysis |

---

## 📚 Detailed Guides

- **[INGESTION.md](INGESTION.md)** — How to load and preprocess data
- **[COMPLETE_GUIDE_TR.md](COMPLETE_GUIDE_TR.md)** — Turkish setup & usage guide
- **[HEALTH_CHECK_REPORT.md](HEALTH_CHECK_REPORT.md)** — Project status & checklist
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** — Technical implementation notes

---

## 📋 Requirements

- **Python:** 3.8+
- **RAM:** 8-16 GB (for embeddings)  
- **Disk:** 5-10 GB (for models & data)
- **GPU:** Optional but recommended (CUDA support for training)

## Project Structure

```
turkish-legal-rag/
├── data/                    # Data storage
│   ├── raw/                # Original documents
│   ├── processed/          # Cleaned and preprocessed data
│   ├── chunked/            # Document chunks for retrieval
│   └── gold/               # Gold-standard evaluation sets
│
├── src/                    # Main source code
│   ├── ingestion/         # Document loading and preprocessing
│   ├── retrieval/         # Embedding-based retrieval
│   ├── reranking/         # Result reranking and filtering
│   ├── llm/               # LLM fine-tuning and generation
│   ├── evaluation/        # Metrics and evaluation scripts
│   └── utils/             # Utilities and helpers
│
├── configs/               # Configuration files (JSON/YAML)
├── scripts/               # Standalone scripts for common tasks
├── notebooks/             # Jupyter notebooks for exploration
├── experiments/           # Experiment logs and results
├── reports/               # Final reports and summaries
├── app/                   # Gradio demo and web interface
│
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Folder Descriptions

- **data/**: Stores all dataset files in different stages of processing
  - `raw/`: Original Turkish legal documents
  - `processed/`: Cleaned documents after preprocessing
  - `chunked/`: Text chunks segmented for retrieval
  - `gold/`: Manual annotations for evaluation

- **src/**: Core codebase organized by functionality
  - `ingestion/`: Loading documents, tokenization, cleaning
  - `retrieval/`: Embedding models, vector indexing (FAISS)
  - `reranking/`: Cross-encoders and ranking algorithms (RankBM25)
  - `llm/`: Fine-tuning scripts (PEFT), inference pipelines
  - `evaluation/`: BLEU, ROUGE, and custom metrics
  - `utils/`: Logging, config loading, common utilities

- **configs/**: Configuration files for different pipeline stages
- **scripts/**: Entry points for training, inference, and evaluation
- **notebooks/**: Exploratory data analysis and development notebooks
- **experiments/**: Tracking of different model versions and results
- **reports/**: Performance summaries and visualizations
- **app/**: Gradio interface for interactive demos

## Quick Start

1. Add your Turkish legal documents to `data/raw/`
2. Run document processing: `python scripts/ingest_demo.py` (test with example data)
3. Or load real datasets: `python scripts/ingest_turkish_datasets.py --hf`
4. Processed data will be in `data/processed/`
5. Launch the demo: `python app/demo.py`
6. Visit the Gradio interface in your browser

**See [INGESTION.md](INGESTION.md) for detailed data ingestion instructions.**

## Development Notes

- Keep configurations in `configs/` directory
- Use `notebooks/` for experimentation before moving to `src/`
- Document all experiments in `experiments/` with timestamps
- Update `reports/` with key findings and metrics

## License

Academic use for university research projects.
