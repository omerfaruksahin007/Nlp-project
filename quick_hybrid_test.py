#!/usr/bin/env python3
"""Quick test of hybrid retrieval system"""

import sys
from pathlib import Path
import json
import pickle

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever

print("✅ Imports successful")

# Test 1: Load BM25 index
print("\n📦 Loading BM25 index...")
sparse_dir = PROJECT_ROOT / 'models' / 'sparse_index'
bm25_file = sparse_dir / 'bm25.pkl'

if not bm25_file.exists():
    print(f"❌ BM25 file not found: {bm25_file}")
    sys.exit(1)

with open(bm25_file, 'rb') as f:
    bm25 = pickle.load(f)
print(f"✅ BM25 index loaded")

# Test 2: Load chunks
print("\n📦 Loading chunks...")
chunks_file = PROJECT_ROOT / 'data' / 'chunked' / 'turkish_law_chunked.jsonl'
chunks = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))
print(f"✅ Loaded {len(chunks)} chunks")

# Test 3: Create sparse retriever
print("\n🔍 Creating sparse retriever...")
sparse_retriever = SparseRetriever()
sparse_retriever.bm25 = bm25
sparse_retriever.chunks = chunks

# Test 4: Test sparse search
query = "Ceza hukuku nedir"
print(f"\n🔍 Testing sparse search with query: '{query}'")
try:
    results = sparse_retriever.search(query, k=5)
    print(f"✅ Got {len(results)} results")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. Score={result.score:.4f}, Text={result.chunk_text[:80]}...")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Quick hybrid test completed!")
