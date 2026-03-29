# 🛠️ PROJECT TROUBLESHOOTING GUIDE

## Quick Verification (Önce Bunu Deneyin!)

To verify the entire system is working:

```bash
# Run the comprehensive diagnostic
python full_diagnostic.py

# This checks:
# 1. All files present
# 2. FAISS index (41,515 vectors)
# 3. Chunks reference (41,515 chunks)
# 4. Component loading (dense, sparse, hybrid)
# 5. Query execution
```

Expected output: `✅ ALL SYSTEMS OPERATIONAL`

---

## Common Issues & Solutions

### Issue #1: "ModuleNotFoundError: No module named 'src'"

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Fix:** Always run scripts from project root with sys.path insert:
```python
import sys
sys.path.insert(0, 'src')
from retrieval.loader import TrainedModelLoader
```

---

### Issue #2: FAISS Index Shows Wrong Vector Count

**Symptom:**
```
Index loaded. Vectors: 27 (WRONG!)
```

**Cause:** Code is loading old pickle format (dense.index) instead of new FAISS binary

**Fix (Check These):**
1. Is `models/retrieval_index/faiss.index` present? (81 MB)
2. Is `dense.py` using FAISS loading first?
   ```python
   # Should try this FIRST:
   faiss_index_path = index_dir / "faiss.index"
   if faiss_index_path.exists():
       index = faiss.read_index(str(faiss_index_path))  # ← 41,515 vectors
   ```

---

### Issue #3: "HybridRetriever.search() got unexpected keyword argument 'k'"

**Symptom:**
```
TypeError: HybridRetriever.search() got an unexpected keyword argument 'k'
```

**Fix:** Use correct parameter names:
```python
# WRONG:
results = hybrid.search(query, k=10)

# CORRECT:
results = hybrid.search(query, k_dense=20, k_sparse=20, k_final=10)
```

---

### Issue #4: Embedding Model Takes Forever to Load

**Symptom:**
```
Loading embedding model...
(waits 2+ minutes)
```

**Expected Behavior:**
- First load: 1-2 minutes (downloading from cache/disk)
- Subsequent loads: ~30 seconds

**Common cause:** Loading from network drive (OneDrive)

**Solutions:**
1. Copy models to local SSD  
2. Use background loading with timeout
3. Pre-load model once at startup

---

### Issue #5: "SparseRetriever.__init__() got unexpected keyword argument 'index_dir'"

**Symptom:**
```
TypeError: SparseRetriever.__init__() got an unexpected keyword argument 'index_dir'
```

**Fix:** SparseRetriever doesn't take index_dir in __init__, use load_bm25():
```python
# WRONG:
sr = SparseRetriever(index_dir='models/sparse_index')

# CORRECT:
sr = SparseRetriever()
sr.load_bm25(index_dir='models/sparse_index', dense_metadata_dir='models/retrieval_index')
```

---

## Performance Tuning

### For Large-Scale Queries
```python
# Default settings (balanced)
results = hybrid.search(query)  
# k_dense=20, k_sparse=20, k_final=10

# Fast (fewer results)
results = hybrid.search(query, k_dense=5, k_sparse=5, k_final=5)

# Comprehensive (more results)
results = hybrid.search(query, k_dense=50, k_sparse=50, k_final=20)
```

### Memory Usage
- FAISS index (in-memory): ~500 MB
- BM25 index (in-memory): ~100 MB
- Model weights: ~514 MB
- **Total:** ~1.1 GB RAM

---

## File Location Reference

**Critical Files (DO NOT MOVE):**
```
models/
├── fine_tuned_embeddings/model.safetensors    ← Embedding model (514 MB)
├── retrieval_index/
│   ├── faiss.index                            ← Dense index (81 MB, 41,515 vectors)
│   ├── chunks_reference.jsonl                 ← Canonical chunk source (24 MB)
│   └── dense.index                            ← OLD (ignore this)
└── sparse_index/
    └── bm25.pkl                               ← BM25 index (19.5 MB)
```

**DO NOT DELETE:**
- `models/retrieval_index/faiss.index` ← This is essential!
- `models/retrieval_index/chunks_reference.jsonl` ← This is essential!
- `models/fine_tuned_embeddings/` ← This is essential!
- `models/sparse_index/bm25.pkl` ← This is essential!

**SAFE TO DELETE:**
- `models/retrieval_index/dense.index` (old pickle, not used)
- `models/retrieval_index/dense_metadata.json` (can be regenerated)
- `models/retrieval_index/dense_config.json` (can be regenerated)
- `models/sparse_index/bm25_metadata.json` (can be regenerated)
- `models/sparse_index/tokenized_chunks.json` (can be regenerated)

---

## Regenerating Corrupted Indexes

### Scenario: FAISS Index Got Corrupted (41,515 → wrong number)

**Step 1:** Delete corrupted file
```powershell
Remove-Item models/retrieval_index/faiss.index
```

**Step 2:** Use Colab notebook to regenerate
- Open: `06_colab_regenerate_faiss.ipynb`
- Run all cells
- Download new `faiss.index` (will be ~85 MB)

**Step 3:** Replace local file
```powershell
Copy-Item Downloads/faiss.index models/retrieval_index/
```

**Step 4:** Verify
```bash
python -c "import faiss; idx = faiss.read_index('models/retrieval_index/faiss.index'); print(idx.ntotal)"
# Should print: 41515
```

---

## Debug Logging

Enable detailed logging to diagnose issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all operations show detailed info
loader = TrainedModelLoader(model_dir='models')
hybrid = loader.load_hybrid_retriever()
results = hybrid.search("your query")
```

Look for:
- `✅` = Success
- `⚠️` = Warning (usually OK)
- `❌` = Error (investigate!)

---

## Performance Benchmarks

**Typical Performance:**
- Model load time: 30-60 seconds
- Query embedding time: 200-500 ms
- Dense search (FAISS k=20): 50-100 ms
- Sparse search (BM25 k=20): 100-200 ms
- RRF fusion: 10-20 ms
- **Total query time:** 400-800 ms

**If slower than benchmarks:**
1. Check if running from OneDrive (much slower)
2. Check disk space (>5 GB free)
3. Check RAM availability (>2 GB free)
4. Check CPU load (should be <50%)

---

## Contact & Support

For issues not listed here:
1. Run `full_diagnostic.py` to get system status
2. Check output of these commands:
   ```bash
   python -c "import faiss; print(faiss.read_index('models/retrieval_index/faiss.index').ntotal)"
   wc -l models/retrieval_index/chunks_reference.jsonl
   ls -lah models/
   ```
3. Check logs in retrieval/*.py files

---

## Last Verified

- Date: 2026-03-29
- Status: ✅ All systems operational
- Test: `full_diagnostic.py` passed
- Components checked: 5/5 ✅
