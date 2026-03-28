"""
PROMPT 3: Document Chunking Module

Chunks long legal documents into 300-token segments with 50-token overlap.
Preserves metadata for retrieval and reranking stages.

Classes:
    DocumentChunker: Main chunking orchestrator
    ChunkProcessor: Process individual chunks
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from transformers import AutoTokenizer
except ImportError:
    # Fallback for testing without transformers
    AutoTokenizer = None


@dataclass
class ChunkStatistics:
    """Statistics about chunking process"""
    total_records_processed: int = 0
    total_chunks_generated: int = 0
    records_with_chunks: int = 0
    records_without_chunks: int = 0
    total_tokens_in_chunks: int = 0
    avg_chunk_length_tokens: float = 0.0
    min_chunk_length: int = 0
    max_chunk_length: int = 0
    chunks_per_record_avg: float = 0.0
    processing_time_seconds: float = 0.0
    error_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class DocumentChunker:
    """
    Main chunking orchestrator for legal documents.
    
    Attributes:
        chunk_size (int): Token count per chunk (300)
        overlap_size (int): Token overlap between chunks (50)
        min_chunk_size (int): Minimum tokens to keep (20)
        tokenizer: HuggingFace tokenizer for token counting
        logger: Logging instance
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        overlap_size: int = 50,
        min_chunk_size: int = 20,
        tokenizer_name: str = "distilbert-base-multilingual-cased"
    ):
        """
        Initialize Chunker.
        
        Args:
            chunk_size: Tokens per chunk
            overlap_size: Token overlap between chunks
            min_chunk_size: Minimum tokens to keep chunk
            tokenizer_name: HuggingFace tokenizer name
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.step_size = chunk_size - overlap_size  # Sliding window step
        
        # Initialize tokenizer
        if AutoTokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
            logging.warning("Transformers not available. Using approximate tokenization.")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.stats = ChunkStatistics()
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        if self.tokenizer is None:
            # Approximate: 1 token ≈ 4 chars in Turkish
            return len(text) // 4
        
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)
    
    def tokenize_text(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            # Approximate tokenization (fallback)
            words = text.split()
            return list(range(len(words)))
        
        encoding = self.tokenizer(text, return_tensors=None, truncation=False)
        return encoding['input_ids']
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            # Approximate decoding (fallback)
            return " ".join(str(i) for i in token_ids)
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def create_sliding_windows(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        """
        Create sliding window indices for chunks.
        
        Algorithm:
            1. Start at position 0
            2. Take chunk_size tokens
            3. Move by step_size (chunk_size - overlap)
            4. Repeat until end of document
            
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of (start, end) tuples for each chunk
        """
        windows = []
        total_tokens = len(token_ids)
        
        # First chunk: 0 to chunk_size
        if total_tokens > 0:
            windows.append((0, min(self.chunk_size, total_tokens)))
        
        # Sliding windows with overlap
        start = self.step_size
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            windows.append((start, end))
            start += self.step_size
        
        return windows
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dicts with 'text' and 'token_count'
        """
        # Tokenize
        token_ids = self.tokenize_text(text)
        
        if len(token_ids) == 0:
            return []
        
        # Create sliding windows
        windows = self.create_sliding_windows(token_ids)
        
        # Extract chunks
        chunks = []
        for start, end in windows:
            chunk_token_ids = token_ids[start:end]
            chunk_token_count = end - start
            
            # Skip very small chunks
            if chunk_token_count < self.min_chunk_size:
                continue
            
            # Decode to text
            chunk_text = self.decode_tokens(chunk_token_ids)
            
            chunks.append({
                'text': chunk_text,
                'token_count': chunk_token_count,
                'start_token': start,
                'end_token': end
            })
        
        return chunks
    
    def chunk_record(self, record: Dict, record_id: str) -> List[Dict]:
        """
        Chunk a single QA record.
        
        Strategy:
            1. Identify text field (question, answer, text)
            2. Chunk the longest field (usually answer)
            3. Prefix each chunk with context (question)
            4. Add metadata from record
            
        Args:
            record: Input record (must have 'question' or 'answer' or 'text')
            record_id: Unique record identifier
            
        Returns:
            List of chunk dicts
        """
        chunks = []
        
        # Identify text fields
        question = record.get('question', '')
        answer = record.get('answer', '')
        text = record.get('text', '')
        
        # Choose what to chunk
        # Priority: answer > text > question
        to_chunk = answer or text or question
        
        if not to_chunk:
            self.logger.warning(f"Record {record_id} has no text to chunk")
            return []
        
        # For QA records: prefix with context
        if question and answer:
            # Combine Q+A for context
            full_text = f"Soru: {question}\n\nCevap: {answer}"
        else:
            full_text = to_chunk
        
        # Chunk the text
        text_chunks = self.chunk_text(full_text)
        
        if not text_chunks:
            self.logger.warning(f"Record {record_id} produced no chunks")
            return []
        
        # Create chunk records with metadata
        for chunk_idx, chunk_info in enumerate(text_chunks, 1):
            chunk_record = {
                'chunk_id': f"{record_id}-chunk-{chunk_idx}",
                'source_record_id': record_id,
                'chunk_text': chunk_info['text'],
                'chunk_length_tokens': chunk_info['token_count'],
                'chunk_position': chunk_idx,
                'total_chunks': len(text_chunks),
                
                # Preserve original metadata
                'law_name': record.get('law_name', ''),
                'article_no': record.get('article_no', ''),
                'section': record.get('section', ''),
                'source': record.get('source', 'unknown'),
                'category': record.get('category', ''),
                
                # Preserve original content for reference
                'metadata': {
                    'original_question': question,
                    'original_answer': answer,
                    'original_text': text,
                    'original_id': record.get('id', ''),
                    'chunk_token_range': [chunk_info['start_token'], chunk_info['end_token']]
                }
            }
            
            chunks.append(chunk_record)
        
        return chunks
    
    def process_jsonl_file(
        self,
        input_path: str,
        output_path: str,
        limit: Optional[int] = None
    ) -> Tuple[int, int, int]:
        """
        Process entire JSONL file and create chunks.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            limit: Max records to process (for testing)
            
        Returns:
            Tuple of (records_processed, chunks_created, errors)
        """
        input_file = Path(input_path)
        output_file = Path(output_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        records_processed = 0
        chunks_created = 0
        errors = 0
        token_count = 0
        
        self.logger.info(f"Starting chunking: {input_file.name} → {output_file.name}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:
                
                for line_num, line in enumerate(infile, 1):
                    if limit and records_processed >= limit:
                        break
                    
                    try:
                        record = json.loads(line.strip())
                        record_id = record.get('id', f'record-{line_num}')
                        
                        # Chunk the record
                        chunks = self.chunk_record(record, record_id)
                        
                        if chunks:
                            # Write chunks to output
                            for chunk in chunks:
                                outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                                chunks_created += 1
                                token_count += chunk['chunk_length_tokens']
                            
                            self.stats.records_with_chunks += 1
                        else:
                            self.stats.records_without_chunks += 1
                        
                        records_processed += 1
                        
                        if records_processed % 1000 == 0:
                            self.logger.info(
                                f"Processed {records_processed} records, "
                                f"created {chunks_created} chunks"
                            )
                    
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Line {line_num}: Invalid JSON - {e}")
                        errors += 1
                    except Exception as e:
                        self.logger.error(f"Line {line_num}: {e}")
                        errors += 1
        
        except IOError as e:
            self.logger.error(f"File I/O error: {e}")
            raise
        
        # Update statistics
        self.stats.total_records_processed = records_processed
        self.stats.total_chunks_generated = chunks_created
        self.stats.total_tokens_in_chunks = token_count
        self.stats.error_count = errors
        
        if chunks_created > 0:
            self.stats.avg_chunk_length_tokens = token_count / chunks_created
            self.stats.chunks_per_record_avg = chunks_created / max(records_processed, 1)
        
        self.logger.info(
            f"Chunking complete: {records_processed} records → "
            f"{chunks_created} chunks ({token_count} total tokens)"
        )
        
        return records_processed, chunks_created, errors
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.jsonl"
    ) -> Dict:
        """
        Process all JSONL files in a directory.
        
        Args:
            input_dir: Input directory (data/processed/)
            output_dir: Output directory (data/chunked/)
            pattern: File pattern to match (*.jsonl)
            
        Returns:
            Summary statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_records = 0
        total_chunks = 0
        total_errors = 0
        
        # Find all matching files
        files = sorted(input_path.glob(pattern))
        
        if not files:
            self.logger.warning(f"No files matching {pattern} in {input_dir}")
            return {'files_processed': 0, 'total_records': 0, 'total_chunks': 0}
        
        self.logger.info(f"Found {len(files)} files to process")
        
        for input_file in files:
            output_file = output_path / f"{input_file.stem}_chunked.jsonl"
            
            self.logger.info(f"Processing: {input_file.name}")
            
            try:
                records, chunks, errors = self.process_jsonl_file(
                    str(input_file),
                    str(output_file)
                )
                total_records += records
                total_chunks += chunks
                total_errors += errors
            except Exception as e:
                self.logger.error(f"Failed to process {input_file.name}: {e}")
                total_errors += 1
        
        return {
            'files_processed': len(files),
            'total_records': total_records,
            'total_chunks': total_chunks,
            'total_errors': total_errors,
            'avg_chunks_per_record': total_chunks / max(total_records, 1)
        }


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('chunker')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger
