# PROMPT 4: DENSE RETRIEVAL - HOW TO RUN

## 🎯 3 Yoldan Birini Seç

### YOL 1: Jupyter Notebook (EN KOLAY - Recommended)
**Avantajlar:** Step-by-step, görsel output, interactive, educational

```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
jupyter notebook notebooks/03_dense_retrieval.ipynb
```

**Notebook'ta çalıştırılacaklar:**
1. Cell 1: Markdown (Konsept)
2. Cell 2: Setup (Paths)
3. Cell 3: Load chunks from Prompt 3
4. Cell 4: Embedding demo (sample texts)
5. Cell 5: Initialize DenseRetriever
6. **Cell 6: MAIN** - Embed all 47.2k chunks (~60-90s CPU)
7. Cell 7: Create FAISS index (~5s)
8. Cell 8: Test search (5 sample queries)
9. Cell 9: Batch search + DataFrame
10. Cell 10: Generate report + statistics

**Expected output:**
```
✅ Embeddings: (47234, 384) shape = 72.5 MB
✅ Index: 47,234 vectors in FAISS
✅ Search: ~35-50ms per query
✅ Report: Saved to models/retrieval_index/
```

**Time:** ~2-3 minutes total (mostly embedding)

---

### YOL 2: Production Script (HIZLI)
**Avantajlar:** One-liner, production-grade, no notebook overhead

```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project

# Default run (build index from chunks)
python scripts/create_dense_index.py

# With custom model
python scripts/create_dense_index.py --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Force rebuild (ignore existing index)
python scripts/create_dense_index.py --force-rebuild

# Log to file
python scripts/create_dense_index.py --log-file dense_build.log
```

**Output:**
```
=======================================================================
PROMPT 4: CREATE DENSE RETRIEVAL INDEX
=======================================================================

Configuration:
  Chunks file: c:\...\data\chunked\turkish_law_chunked.jsonl
  Output dir: c:\...\models\retrieval_index
  Model: distiluse-base-multilingual-cased-v2
  Device: cpu

Loading embedding model...
✅ Model loaded. Dimension: 384

Embedding 47234 chunks...
✅ Embeddings created: (47234, 384)

Creating FAISS index...
✅ Index created with 47234 vectors

✅ Index creation complete!
Index location: models/retrieval_index

Files:
  dense.index: 72.00 MB
  dense_metadata.json: 15.23 MB
  dense_config.json: 0.45 MB

Next step: scripts/test_dense_retrieval.py
```

**Time:** ~1-2 minutes (mostly embedding)

---

### YOL 3: Test Existing Index (VERIFY)
**Avantajlar:** Quick verification, no rebuilding

```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project

# Test with default queries
python scripts/test_dense_retrieval.py

# Test with custom query
python scripts/test_dense_retrieval.py --query "Kasten adam öldürme"

# Get top-10 results
python scripts/test_dense_retrieval.py --top-k 10
```

**Output:**
```
=======================================================================
PROMPT 4: TEST DENSE RETRIEVAL
=======================================================================

Loading FAISS index from models/retrieval_index...
✅ Index loaded: 47,234 vectors
   Dimension: 384
   Model: distiluse-base-multilingual-cased-v2

Testing search with 5 queries (top-5)...

Query: Kasten adam öldürme suçu nedir?
Results: 5 chunks

  Rank 1:
    Score: 0.9425
    Law: Türk Ceza Kanunu Madde 81
    Text: Soru: Kasten adam öldürme suçu nedir?...

  Rank 2:
    Score: 0.8312
    Law: Türk Ceza Kanunu Madde 81
    Text: Cevap: ...ağırlaştırıcı sebepler...

  [3 more ranks...]

Query: Taksir halinde ölüm cezası
Results: 5 chunks
[...]

✅ Sample results exported: models/retrieval_index/sample_search_results.json

Ready for Prompt 5: Reranking with Cross-Encoder
```

**Time:** ~30 seconds (load + test)

---

## 📊 SONRA NEYİ KONTROL ET?

### Output Files
```
models/retrieval_index/
├─ dense.index                    (72 MB - FAISS index)
├─ dense_metadata.json            (15 MB - Chunk metadata)
├─ dense_config.json              (small - Configuration)
├─ retrieval_report.json          (Performance stats)
└─ sample_search_results.json     (Sample output)
```

### Kontrolü
```bash
# Check file sizes
ls -lh models/retrieval_index/

# View report
python -c "import json; print(json.dumps(json.load(open('models/retrieval_index/retrieval_report.json')), indent=2))"

# Load and test manually
python
>>> from src.retrieval.dense import DenseRetriever
>>> retriever = DenseRetriever()
>>> retriever.load_index()
>>> results = retriever.search("Madde 81", k=5)
>>> for r in results: print(f"{r.rank}. {r.score:.4f} - {r.chunk_text[:50]}")
```

---

## ⚠️ ÇIKMAZLAR VE ÇÖZÜMLER

### Problem 1: "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
# Large download (~500 MB model cache)
```

### Problem 2: "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu
# Or GPU version:
pip install faiss-gpu
```

### Problem 3: "No JSONL file found in data/chunked"
- Çalıştır: `python scripts/chunk_documents.py` (Prompt 3)
- Sonra tekrar: `python scripts/create_dense_index.py`

### Problem 4: "CUDA out of memory" (if using GPU)
```bash
# Use CPU instead
python scripts/create_dense_index.py --device cpu

# Or reduce batch size in code
```

### Problem 5: "Embedding very slow" (taking >5 minutes)
- **Normal:** 60-90s on CPU is expected
- **GPU:** Should be 10-20s
- **Slow:** May be CPU-throttled. Check:
  ```bash
  # Windows PowerShell
  Get-Process | Where-Object {$_.ProcessName -eq "python"} | Select-Object Name, CPU
  ```

### Problem 6: "Search returns empty results"
```python
# Check if index loaded correctly
>>> from src.retrieval.dense import DenseRetriever
>>> r = DenseRetriever()
>>> r.load_index()
>>> print(r.index.ntotal)  # Should be 47234
>>> results = r.search("test", k=5)
>>> print(len(results))  # Should be 5
```

---

## 🎯 QUICK START (Copy-Paste)

### Fastest Way: Notebook
```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
jupyter notebook notebooks/03_dense_retrieval.ipynb
# Kernel → Run All
```

### Fastest Way: Script
```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
python scripts/create_dense_index.py
```

### Fastest Way: Verify
```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
python scripts/test_dense_retrieval.py
```

---

## 📈 BEKLENEN ÇIKTI

### Step 1: Chunks Loaded
```
✅ Total chunks loaded: 47,234
✅ Sample chunk:
   ID: 550e8400-e29b-41d4-a716-chunk-1
   Tokens: 298
   Text: "Soru: Kasten adam öldürme..."
```

### Step 2: Embedding Finished
```
✅ Embeddings created: (47234, 384)
   Data type: float32
   Memory: 72.33 MB
   Time: 74.82s
   Speed: 630 chunks/sec
```

### Step 3: Index Created
```
✅ Index created:
   Index type: IndexFlatL2
   Total vectors: 47,234
   Dimension: 384
   Time: 2.34s
   
📁 Index files:
   dense.index: 72.00 MB
   dense_metadata.json: 15.23 MB
   dense_config.json: 0.05 MB
```

### Step 4: Search Tested
```
Query 1: "Kasten adam öldürme suçu nedir?"
  Time: 34ms
  Results: 5 chunks
  
Query 2: "Taksir halinde ölüm cezası"
  Time: 38ms
  Results: 5 chunks

✅ Search Performance:
   Avg time: 36.40ms
   Min time: 22.45ms
   Max time: 51.23ms
   QPS: 27.5 queries/sec
```

---

## ✨ AFTER DENSE RETRIEVAL

Prompt 4 bitince:
```
✅ 47k chunks embedded (384-dim vectors)
✅ FAISS index created (~72 MB)
✅ Search working (<50ms/query)
✅ Top-20 retrieval accurate

→ Ready for PROMPT 5: BM25 + Hybrid Retrieval
   - Tokenize chunks for BM25
   - Build sparse index
   - Fuse dense + sparse
   - Get top-10 after fusion
```

---

## 🔄 ITERATIVE TESTING

Different embedding models:

```bash
# Test 1: Small model (faster)
python scripts/create_dense_index.py \
  --model "sentence-transformers/paraphrase-MiniLM-L6-v2" \
  --force-rebuild

# Test 2: Larger model (better quality)
python scripts/create_dense_index.py \
  --model "sentence-transformers/distiluse-base-multilingual-cased-v2" \
  --force-rebuild

# Test 3: Multilingual (120+ languages)
python scripts/create_dense_index.py \
  --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  --force-rebuild

# Compare results for same query
python scripts/test_dense_retrieval.py --query "Madde 81"
```

---

## 📝 KONTROL LİSTESİ

Dense retrieval başarılı olursa:

- [ ] Chunks loaded (47,234)
- [ ] Embedding model initialized
- [ ] All chunks embedded in <2 minutes
- [ ] FAISS index created (~72 MB)
- [ ] Index files saved to models/retrieval_index/
- [ ] Search latency <50ms per query
- [ ] Top-5 results make sense
- [ ] Metadata preserved (law_name, article_no)
- [ ] Report generated (retrieval_report.json)
- [ ] No OOM errors
- [ ] Reproducible (same results for same query)

---

## 🎓 ÖĞRENME NOKTASI

Prompt 4 tamamlanırsa, şunu öğrendin:

1. ✅ **Embedding:** Text → Dense vector (384-dim)
2. ✅ **Similarity:** Vektörler arasındaki benzerlik (cosine/L2)
3. ✅ **FAISS:** Fast approximate similarity search
4. ✅ **Indexing:** 47k vectors indexed in <100MB
5. ✅ **Scale:** Can handle millions with IVF
6. ✅ **Search:** Query → vector → top-k chunks (<50ms)

Hazır mısın **Prompt 5: BM25 + Hybrid Retrieval** için? 🚀
