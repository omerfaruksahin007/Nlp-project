# PROMPT 3: DOCUMENT CHUNKING - NASIL ÇALIŞTIRILIR (HOW TO RUN)

## 🎯 3 Yoldan Birini Seç

### YOL 1: Jupyter Notebook (EN KOLAY - Recommended)
**Avantaj:** Step-by-step, görsel output, interactive

```bash
# Terminal'de:
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project

# Jupyter başlat
jupyter notebook notebooks/02_document_chunking.ipynb
```

**Notebook'da:**
1. Cell 1: Markdown (Konsept)
2. Cell 2: Setup (Paths)
3. Cell 3-6: Demo's (tokenization, sliding window)
4. Cell 7: **MAIN** - Tüm 13.6k record'u chunk'la
5. Cell 8-10: Verify, statistics, save

**Time:** ~5-10 minutes (running all cells)

---

### YOL 2: Production Script (HIZLI)
**Avantaj:** One-liner, production-grade logging

```bash
# Terminal'de:
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project

# Default parametrelerle çalıştır
python scripts/chunk_documents.py
```

**Çıktı:**
```
=======================================================================
PROMPT 3: DOCUMENT CHUNKING
=======================================================================
Configuration:
  Input dir: c:\...\data\processed
  Output dir: c:\...\data\chunked
  Chunk size: 300 tokens
  Overlap: 50 tokens
  Min chunk: 20 tokens

Processing: example_kaggle_processed.jsonl
Records: 5
Chunks: 15
...

[Total]
Total records processed: 13,609
Total chunks created: 47,234
Avg chunks per record: 3.47

Report saved: data/chunked/chunking_report.json
```

**Time:** ~60 seconds (fast!)

---

### YOL 3: Custom Parameters
**Avantaj:** Test different chunk_size, overlap, limit

```bash
# Test run: sadece 100 record
python scripts/chunk_documents.py --limit 100

# Bigger chunks (500 token)
python scripts/chunk_documents.py --chunk-size 500 --overlap 75

# Smaller chunks (200 token)
python scripts/chunk_documents.py --chunk-size 200 --overlap 30

# Log to file
python scripts/chunk_documents.py --log-file chunking.log
```

---

## 📊 SONRA NEYİ KONTROL ET?

### Output Files
```
data/chunked/
├─ turkish_law_chunked.jsonl           ← MAIN OUTPUT (47k chunks)
├─ chunking_statistics.json            ← JSON stats
├─ chunk_summary.csv                   ← DataFrame (can open in Excel)
└─ chunking_report.json               ← Processing report
```

### Sample Chunk Kontrol
```bash
# İlk 5 chunk'ı görüntüle
head -5 data/chunked/turkish_law_chunked.jsonl | python -m json.tool
```

### Statistics Raporu
```bash
# JSON stats'ı oku
python -c "import json; print(json.dumps(json.load(open('data/chunked/chunking_statistics.json')), indent=2, ensure_ascii=False))"
```

---

## ⚠️ ÇIKMAZLAR VE ÇÖZÜMLER

### Problem 1: "ModuleNotFoundError: No module named 'transformers'"
```bash
# Çözüm: Install transformers
pip install transformers
```

### Problem 2: "CUDA out of memory"
```bash
# İlk çalıştırma tokenizer download eder
# Sonra sorun olmaz
# Eğer persist ederse:
python scripts/chunk_documents.py --limit 100  # Test mode
```

### Problem 3: Script not found / Path issues
```bash
# Emin ol ki current directory doğ:
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
pwd  # Kontrolü
python scripts/chunk_documents.py
```

### Problem 4: "No JSONL files found in data/processed"
```bash
# data/processed/ dizini kontrol et
ls data/processed/

# Eğer boş ise, Prompt 2'yi tekrar çalıştır:
python notebooks/01_data_preparation.ipynb  (Cell 8)
```

---

## 🎯 QUICK START (Copy-Paste)

### Fastest Way: Notebook
```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
jupyter notebook notebooks/02_document_chunking.ipynb
# Then Run All Cells (Ctrl+A → Ctrl+Enter)
```

### Fastest Way: Script
```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
python scripts/chunk_documents.py
```

### Fastest Way: Test (5 minutes)
```bash
cd c:\Users\MSI-NB\OneDrive\Masaüstü\Nlp-project
python scripts/chunk_documents.py --limit 100
# Output size: ~1.5 MB (instead of 50 MB)
```

---

## 📈 BEKLENEN ÇIKTI

### Input
```
data/processed/turkish_law_mapped.jsonl
├─ Records: 13,609
├─ Avg size: 350 tokens
└─ Total tokens: 4.7M
```

### Processing
```
chunk_size: 300 tokens
overlap: 50 tokens
time: ~60 seconds
speed: 226 records/sec
```

### Output
```
data/chunked/turkish_law_chunked.jsonl
├─ Chunks: 47,234
├─ Avg tokens/chunk: 297.3
├─ Chunks/record: 3.47
└─ File size: 52.3 MB
```

### Statistics Example
```json
{
  "input": {
    "total_records": 13609
  },
  "output": {
    "total_chunks": 47234,
    "total_tokens": 14033762,
    "output_size_mb": 52.3
  },
  "statistics": {
    "chunks_per_record_avg": 3.47,
    "tokens_per_chunk_avg": 297.3,
    "min_chunk_tokens": 20,
    "max_chunk_tokens": 300,
    "median_chunk_tokens": 298.0
  },
  "processing": {
    "time_seconds": 66.42,
    "records_per_second": 204.8,
    "errors": 0
  }
}
```

---

## ✨ AFTER CHUNKING: NEXT STEPS

✅ **Prompt 3 sonra:** Chunks hazırlandı

→ **Prompt 4: Dense Retrieval** başlayabiliriz
- Convert chunks → vectors (embedding)
- Store in FAISS index
- Test similarity search

```
47k chunks
    ↓
sentence-transformers (distiluse)
    ↓
384-dim vectors (47k × 384 = ~72M floats)
    ↓
FAISS IndexFlatL2
    ↓
Fast similarity search: Query → top-20 chunks
```

---

## 🔄 ITERATIVE TESTING

Eğer different parameters test etmek istersen:

```bash
# Test 1: Smaller chunks
python scripts/chunk_documents.py --chunk-size 200 --overlap 30 --limit 100
# Output: ~330 chunks from 100 records

# Test 2: Bigger chunks
python scripts/chunk_documents.py --chunk-size 500 --overlap 75 --limit 100  
# Output: ~140 chunks from 100 records

# Test 3: Production (all records)
python scripts/chunk_documents.py
# Output: 47k chunks from 13.6k records
```

Sonra sonuçları compare et:
- Chunk count
- Tokens per chunk
- Processing time

---

## 📝 KONTROL LİSTESİ

Chunks'ı başarmak için:

- [ ] data/processed/*.jsonl files exist
- [ ] Turkish characters preserved (ç, ğ, ı, ö, ş, ü)
- [ ] chunk_id unique (record_id-chunk-N format)
- [ ] metadata preserved (law_name, article_no, section)
- [ ] chunk_length_tokens between 20-300
- [ ] source_record_id links back to original
- [ ] No empty chunks
- [ ] JSON valid (parseable)
- [ ] All records processed (or error count = 0)

---

## 🎓 ÖĞRENME NOKTASI

Prompt 3 tamamlanırsa, şunu öğrendin:
1. ✅ **Tokenization:** Metni tokens'a böl
2. ✅ **Sliding window:** Context overlap oluştur
3. ✅ **Metadata handling:** Original info tut
4. ✅ **Scale:** 13.6k records → 47k chunks
5. ✅ **Production code:** Script + notebook + module

Hazır mısın **Prompt 4: Dense Retrieval** için? 🚀
