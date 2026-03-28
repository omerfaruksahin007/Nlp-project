#!/usr/bin/env python3
"""
Load Turkish law dataset from CSV and save as processed JSONL
"""

import pandas as pd
import json
from pathlib import Path
import logging
import sys
import uuid
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_csv_and_convert(csv_path: str, output_jsonl: str) -> int:
    """
    Load CSV and convert to processed JSONL format
    
    Expected CSV columns:
    - question or soru
    - answer or cevap  
    - law_name or kanun_adi
    - article_no or madde_no
    """
    
    csv_file = Path(csv_path)
    output_file = Path(output_jsonl)
    
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 0
    
    logger.info(f"Loading CSV from: {csv_path}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        logger.info(f"✅ Loaded {len(df)} rows from CSV")
        logger.info(f"   Columns: {list(df.columns)}")
        
        # Detect column names (might be Turkish or English)
        cols = {col.lower(): col for col in df.columns}
        
        question_col = cols.get('question') or cols.get('soru') or cols.get('q')
        answer_col = cols.get('answer') or cols.get('cevap') or cols.get('answer_text')
        law_col = cols.get('law_name') or cols.get('kanun_adi') or cols.get('law')
        article_col = cols.get('article_no') or cols.get('madde_no') or cols.get('article')
        section_col = cols.get('section') or cols.get('bolum') or cols.get('category')
        
        logger.info(f"   Detected columns:")
        logger.info(f"     Question: {question_col}")
        logger.info(f"     Answer: {answer_col}")
        logger.info(f"     Law: {law_col}")
        logger.info(f"     Article: {article_col}")
        logger.info(f"     Section: {section_col}")
        
        # Process and convert
        records = []
        skipped = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract fields
                question = str(row[question_col]).strip() if question_col else ""
                answer = str(row[answer_col]).strip() if answer_col else ""
                law_name = str(row[law_col]).strip() if law_col else "Bilinmiyor"
                article_no = str(row[article_col]).strip() if article_col else ""
                section = str(row[section_col]).strip() if section_col else ""
                
                # Skip if no question or answer
                if not question or not answer:
                    skipped += 1
                    continue
                
                # Clean values
                question = question.replace('\n', ' ').strip()
                answer = answer.replace('\n', ' ').strip()
                
                # Create record with unified schema
                record = {
                    'id': str(uuid.uuid4()),
                    'question': question,
                    'answer': answer,
                    'law_name': law_name,
                    'article_no': article_no,
                    'section': section,
                    'source': 'turkish_law_dataset',
                    'metadata': {
                        'csv_row': idx,
                        'original_columns': list(row.index.tolist())
                    }
                }
                
                records.append(record)
                
            except Exception as e:
                logger.warning(f"Skipped row {idx}: {e}")
                skipped += 1
                continue
        
        # Save as JSONL
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"\n✅ Successfully processed:")
        logger.info(f"   Total: {len(df)} rows")
        logger.info(f"   Converted: {len(records)} records")
        logger.info(f"   Skipped: {skipped} rows")
        logger.info(f"   Output: {output_file}")
        logger.info(f"   Size: {output_file.stat().st_size / (1024**2):.2f} MB")
        
        return len(records)
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}", exc_info=True)
        return 0

def main():
    """Main entry point"""
    csv_path = "data/raw/turkish_law_dataset.csv"
    output_path = "data/processed/turkish_law.jsonl"
    
    logger.info("="*80)
    logger.info("PROMPT 2: Data Preparation (CSV → JSONL)")
    logger.info("="*80 + "\n")
    
    count = load_csv_and_convert(csv_path, output_path)
    
    if count > 0:
        logger.info(f"\n✅ PROMPT 2 COMPLETE: {count} records processed")
        return 0
    else:
        logger.error(f"\n❌ PROMPT 2 FAILED: No records processed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
