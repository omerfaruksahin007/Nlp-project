#!/usr/bin/env python3
"""
PROMPT 11: Main Evaluation Framework Runner
Orchestrates all evaluation metrics and generates report
"""

import json
import csv
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Dict, Optional
from datetime import datetime

# Add src to path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from evaluation.metrics import RetrievalEvaluator, QAEvaluator, RetrievalMetrics, QAMetrics
from evaluation.hallucination import HallucinationDetector, HallucinationType
from evaluation.citations import CitationEvaluator

@dataclass
class EvaluationResult:
    """Complete evaluation for one test case"""
    question: str
    gold_answer: str
    predicted_answer: str
    retrieved_doc_ids: List[str]
    gold_doc_ids: List[str]
    gold_citations: List[str]
    
    # Metrics
    retrieval_metrics: dict
    qa_metrics: dict
    citation_metrics: dict
    hallucination_analysis: dict

class RAGEvaluationFramework:
    """Complete evaluation framework"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[EvaluationResult] = []
    
    def evaluate_single(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        retrieved_doc_ids: List[str],
        gold_doc_ids: List[str],
        retrieved_docs: List[str],
        gold_citations: List[str],
        model_confidence: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single Q&A + retrieval instance
        
        Args:
            question: Input question
            gold_answer: Ground truth answer
            predicted_answer: Model's answer
            retrieved_doc_ids: IDs of retrieved documents
            gold_doc_ids: IDs of relevant documents (gold)
            retrieved_docs: Actual retrieved document texts
            gold_citations: Expected citations
            model_confidence: Model's confidence level (low/high)
            
        Returns:
            EvaluationResult with all metrics
        """
        
        # 1. RETRIEVAL METRICS
        retrieval_metrics = RetrievalEvaluator.evaluate(retrieved_doc_ids, gold_doc_ids)
        retrieval_dict = asdict(retrieval_metrics)
        
        # 2. QA METRICS
        qa_metrics = QAEvaluator.evaluate(predicted_answer, gold_answer)
        qa_dict = asdict(qa_metrics)
        
        # 3. CITATION METRICS
        predicted_citations = list(
            CitationEvaluator.extract_cited_articles(predicted_answer).keys()
        )
        citation_eval = CitationEvaluator.evaluate(predicted_citations, gold_citations)
        citation_dict = asdict(citation_eval)
        
        # 4. HALLUCINATION ANALYSIS
        halluc = HallucinationDetector.analyze(
            predicted_answer, retrieved_docs, gold_answer, model_confidence
        )
        halluc_dict = {
            'type': halluc.hallucination_type.value,
            'confidence': halluc.confidence,
            'explanation': halluc.explanation,
            'supporting_evidence_count': len(halluc.supporting_evidence or []),
            'unsupported_claims_count': len(halluc.unsupported_claims or [])
        }
        
        result = EvaluationResult(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
            retrieved_doc_ids=retrieved_doc_ids,
            gold_doc_ids=gold_doc_ids,
            gold_citations=gold_citations,
            retrieval_metrics=retrieval_dict,
            qa_metrics=qa_dict,
            citation_metrics=citation_dict,
            hallucination_analysis=halluc_dict
        )
        
        self.results.append(result)
        return result
    
    def save_csv(self, filename: str = "evaluation_results.csv"):
        """Save results as CSV"""
        if len(self.results) == 0:
            print("[!] No results to save")
            return
        
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'question',
                'retrieval_recall@5',
                'retrieval_recall@10',
                'retrieval_mrr',
                'retrieval_ndcg@5',
                'qa_exact_match',
                'qa_token_f1',
                'qa_bleu',
                'qa_rouge_l',
                'citation_precision',
                'citation_recall',
                'citation_f1',
                'hallucination_type',
                'hallucination_confidence'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'question': result.question[:100],  # Truncate for CSV
                    'retrieval_recall@5': result.retrieval_metrics['recall_at_5'],
                    'retrieval_recall@10': result.retrieval_metrics['recall_at_10'],
                    'retrieval_mrr': result.retrieval_metrics['mrr'],
                    'retrieval_ndcg@5': result.retrieval_metrics['ndcg_at_5'],
                    'qa_exact_match': result.qa_metrics['exact_match'],
                    'qa_token_f1': result.qa_metrics['token_f1'],
                    'qa_bleu': result.qa_metrics['bleu'],
                    'qa_rouge_l': result.qa_metrics['rouge_l'],
                    'citation_precision': result.citation_metrics['precision'],
                    'citation_recall': result.citation_metrics['recall'],
                    'citation_f1': result.citation_metrics['f1'],
                    'hallucination_type': result.hallucination_analysis['type'],
                    'hallucination_confidence': result.hallucination_analysis['confidence']
                }
                writer.writerow(row)
        
        print(f"[OK] CSV saved: {csv_path}")
    
    def save_json(self, filename: str = "evaluation_results.json"):
        """Save results as JSON"""
        if len(self.results) == 0:
            print("[!] No results to save")
            return
        
        json_path = self.output_dir / filename
        
        # Convert results to JSON-serializable format
        results_list = []
        for result in self.results:
            results_list.append({
                'question': result.question,
                'gold_answer': result.gold_answer,
                'predicted_answer': result.predicted_answer,
                'retrieval_metrics': result.retrieval_metrics,
                'qa_metrics': result.qa_metrics,
                'citation_metrics': result.citation_metrics,
                'hallucination_analysis': result.hallucination_analysis
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] JSON saved: {json_path}")
    
    def generate_markdown_report(self, filename: str = "EVALUATION_REPORT.md"):
        """Generate comprehensive markdown report"""
        if len(self.results) == 0:
            print("[!] No results to report")
            return
        
        md_path = self.output_dir / filename
        
        # Calculate aggregate metrics
        avg_retrieval = {
            'recall_at_5': sum(r.retrieval_metrics['recall_at_5'] for r in self.results) / len(self.results),
            'recall_at_10': sum(r.retrieval_metrics['recall_at_10'] for r in self.results) / len(self.results),
            'mrr': sum(r.retrieval_metrics['mrr'] for r in self.results) / len(self.results),
            'ndcg_at_5': sum(r.retrieval_metrics['ndcg_at_5'] for r in self.results) / len(self.results),
            'ndcg_at_10': sum(r.retrieval_metrics['ndcg_at_10'] for r in self.results) / len(self.results),
        }
        
        avg_qa = {
            'exact_match': sum(r.qa_metrics['exact_match'] for r in self.results) / len(self.results),
            'token_f1': sum(r.qa_metrics['token_f1'] for r in self.results) / len(self.results),
            'bleu': sum(r.qa_metrics['bleu'] for r in self.results) / len(self.results),
            'rouge_l': sum(r.qa_metrics['rouge_l'] for r in self.results) / len(self.results),
        }
        
        avg_citation = {
            'precision': sum(r.citation_metrics['precision'] for r in self.results) / len(self.results),
            'recall': sum(r.citation_metrics['recall'] for r in self.results) / len(self.results),
            'f1': sum(r.citation_metrics['f1'] for r in self.results) / len(self.results),
        }
        
        # Count hallucinations
        halluc_counts = {}
        for result in self.results:
            htype = result.hallucination_analysis['type']
            halluc_counts[htype] = halluc_counts.get(htype, 0) + 1
        
        # Generate report
        report = f"""# RAG Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total test cases: {len(self.results)}
- Report saved to: {md_path}

## 1. RETRIEVAL METRICS

| Metric | Score |
|--------|-------|
| Recall@5 | {avg_retrieval['recall_at_5']:.3f} |
| Recall@10 | {avg_retrieval['recall_at_10']:.3f} |
| Mean Reciprocal Rank (MRR) | {avg_retrieval['mrr']:.3f} |
| nDCG@5 | {avg_retrieval['ndcg_at_5']:.3f} |
| nDCG@10 | {avg_retrieval['ndcg_at_10']:.3f} |

### Interpretation
- **Recall@5/10**: Fraction of relevant docs found in top-5/10
- **MRR**: How high does first relevant doc rank?
- **nDCG**: Ranking quality considering positions

## 2. QA ANSWER METRICS

| Metric | Score |
|--------|-------|
| Exact Match | {avg_qa['exact_match']:.3f} |
| Token F1 | {avg_qa['token_f1']:.3f} |
| BLEU | {avg_qa['bleu']:.3f} |
| ROUGE-L | {avg_qa['rouge_l']:.3f} |

### Interpretation
- **Exact Match**: Perfect match with gold (0-1)
- **Token F1**: Token overlap with gold answer
- **BLEU**: N-gram precision (penalizes brevity)
- **ROUGE-L**: LCS-based F1 (sequence matching)

## 3. CITATION METRICS

| Metric | Score |
|--------|-------|
| Citation Precision | {avg_citation['precision']:.3f} |
| Citation Recall | {avg_citation['recall']:.3f} |
| Citation F1 | {avg_citation['f1']:.3f} |

### Interpretation
- **Precision**: Fraction of cited sources that are correct
- **Recall**: Fraction of relevant citations mentioned
- **F1**: Harmonic mean

## 4. HALLUCINATION ANALYSIS

"""
        
        # Hallucination breakdown
        report += "### Hallucination Distribution\n\n"
        for htype, count in sorted(halluc_counts.items()):
            pct = 100 * count / len(self.results)
            report += f"- **{htype}**: {count} cases ({pct:.1f}%)\n"
        
        report += """

### Hallucination Types
- **CORRECT**: Properly supported answer
- **PARTIALLY_SUPPORTED**: Some support but incomplete
- **UNSUPPORTED_CLAIM**: Claims not in retrieved docs
- **WRONG_CITATION**: Citations don't match content
- **NO_ANSWER_DESPITE_EVIDENCE**: Evidence available but unused

## 5. DETAILED RESULTS

"""
        
        # Per-case results
        for i, result in enumerate(self.results, 1):
            report += f"\n### Test Case {i}\n\n"
            report += f"**Question:** {result.question}\n\n"
            report += f"**Gold Answer:** {result.gold_answer[:200]}...\n\n"
            report += f"**Predicted:** {result.predicted_answer[:200]}...\n\n"
            
            report += "**Metrics:**\n"
            report += f"- Retrieval: Recall@5={result.retrieval_metrics['recall_at_5']:.2f}, MRR={result.retrieval_metrics['mrr']:.2f}\n"
            report += f"- QA: EM={result.qa_metrics['exact_match']:.2f}, F1={result.qa_metrics['token_f1']:.2f}\n"
            report += f"- Hallucination: {result.hallucination_analysis['type']} (conf: {result.hallucination_analysis['confidence']:.2f})\n"
        
        report += """

## 6. RECOMMENDATIONS

1. **Low Retrieval Metrics?**
   - Improve document chunking
   - Add more training data
   - Tune similarity thresholds

2. **Low QA Metrics?**
   - Use better language model for generation
   - Add better prompting
   - Fine-tune on domain-specific data

3. **Citation Issues?**
   - Improve citation extraction
   - Add citation validation
   - Include citation in training

4. **High Hallucination Rate?**
   - Increase retrieval k
   - Add grounding checks
   - Use confidence calibration

## 7. METHODOLOGY

This evaluation framework uses:
- **Retrieval Metrics**: Standard IR metrics (Recall, MRR, nDCG)
- **QA Metrics**: Token F1, BLEU, ROUGE (standard NLG metrics)
- **Citation Eval**: Precision/Recall against ground truth citations
- **Hallucination**: Claim coverage + citation validity analysis

All metrics are transparent and reproducible.

---
Report generated by PROMPT 11 Evaluation Framework
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Markdown report saved: {md_path}")
    
    def print_summary(self):
        """Print summary to console"""
        if len(self.results) == 0:
            print("[!] No results")
            return
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Average metrics
        avg_retrieval_recall5 = sum(r.retrieval_metrics['recall_at_5'] for r in self.results) / len(self.results)
        avg_qa_f1 = sum(r.qa_metrics['token_f1'] for r in self.results) / len(self.results)
        avg_citation_f1 = sum(r.citation_metrics['f1'] for r in self.results) / len(self.results)
        
        print(f"\nTest Cases Evaluated: {len(self.results)}")
        print(f"\nRetrieval - Recall@5: {avg_retrieval_recall5:.3f}")
        print(f"QA - Token F1: {avg_qa_f1:.3f}")
        print(f"Citations - F1: {avg_citation_f1:.3f}")
        
        # Hallucination breakdown
        halluc_counts = {}
        for result in self.results:
            htype = result.hallucination_analysis['type']
            halluc_counts[htype] = halluc_counts.get(htype, 0) + 1
        
        print("\nHallucination Distribution:")
        for htype, count in sorted(halluc_counts.items()):
            pct = 100 * count / len(self.results)
            print(f"  {htype}: {count} ({pct:.0f}%)")
        
        print("\n" + "=" * 80)
