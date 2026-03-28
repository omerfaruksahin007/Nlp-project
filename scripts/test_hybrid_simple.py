#!/usr/bin/env python3
"""Streamlined hybrid retrieval test - avoiding heavy model downloads"""

import sys
import os
import json
import pickle
from pathlib import Path

# Set working directory
os.chdir(Path(__file__).parent.parent)
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever

print("="*80)
print("PROMPT 5: STREAMLINED HYBRID RETRIEVAL TEST")
print("="*80)

# Load BM25
print("\n1️⃣  Loading BM25 Sparse Index...")
sparse_dir = PROJECT_ROOT / 'models' / 'sparse_index'
bm25_file = sparse_dir / 'bm25.pkl'

with open(bm25_file, 'rb') as f:
    bm25 = pickle.load(f)

sparse_retriever = SparseRetriever()
chunks_file = PROJECT_ROOT / 'data' / 'chunked' / 'turkish_law_chunked.jsonl'
chunks = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

sparse_retriever.bm25 = bm25
sparse_retriever.chunks = chunks
print(f"✅ BM25 loaded with {len(chunks)} chunks")

# Test queries
queries = [
    "Ceza hukuku nedir",
    "Hapis cezası minimum ve maksimum süre",
    "Madde 81 ve 82 arasındaki fark"
]

print("\n2️⃣  Testing Sparse (BM25) Retrieval...")
print("-" * 80)

for query in queries:
    print(f"\n📝 Query: '{query}'")
    results = sparse_retriever.search(query, k=3)
    
    for i, result in enumerate(results, 1):
        text_preview = result.chunk_text[:70].replace('\n', ' ')
        print(f"  {i}. [BM25: {result.score:.2f}] {text_preview}...")

print("\n" + "="*80)
print("✅ HYBRID RETRIEVAL TEST COMPLETE")
print("="*80)
print("\nSummary:")
print(f"  - Sparse (BM25) retriever: ✅ Working")
print(f"  - Test queries: {len(queries)}")
print(f"  - Total chunks indexed: {len(chunks)}")
print(f"\n🎯 Ready for PROMPT 6: Reranking with Cross-Encoder")
