# Turkish Legal RAG

## Project Objective

A Turkish legal question answering system based on Retrieval-Augmented Generation (RAG). This system retrieves relevant legal documents and generates contextual answers to user queries in Turkish legal domain.

## Key Features

- **Document Ingestion**: Load and preprocess Turkish legal documents
  - Supports HuggingFace datasets (Renicames/turkish-lawchatbot)
  - Supports Kaggle datasets (batuhankalem/turkishlaw-dataset-for-llm-finetuning)
  - Handles CSV, JSON, JSONL formats
  - Turkish-aware text normalization
- **Semantic Retrieval**: Find relevant documents using embedding-based search
- **Reranking**: Improve retrieval quality with cross-encoder reranking
- **LLM Fine-tuning**: Adapt language models for Turkish legal domain
- **Evaluation**: Measure system performance with standard metrics
- **Interactive Demo**: Gradio-based web interface for testing

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd turkish-legal-rag
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On Linux/Mac
   
   # Or using conda
   conda create -n turkish-legal-rag python=3.10
   conda activate turkish-legal-rag
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import sentence_transformers; import faiss; print('Setup complete!')"
   ```

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
