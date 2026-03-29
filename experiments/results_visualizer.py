#!/usr/bin/env python3
"""
PROMPT 12 - Results Comparison and Visualization
Generates comparison tables and visualizations for ablation studies
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class ResultsComparator:
    """Compares results across experiments"""
    
    def __init__(self, experiments_dir: Path):
        """
        Initialize results comparator
        
        Args:
            experiments_dir: Directory containing experiment results
        """
        self.experiments_dir = Path(experiments_dir)
        self.results = {}
        
    def load_experiment_results(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load results from a single experiment"""
        metrics_file = self.experiments_dir / experiment_name / f"{experiment_name}_metrics.json"
        
        if not metrics_file.exists():
            logger.warning(f"Results not found: {metrics_file}")
            return None
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {metrics_file}: {e}")
            return None
    
    def load_all_results(self) -> None:
        """Load results from all experiments"""
        from experiment_configs import EXPERIMENTS
        
        self.results = {}
        for exp_key, config in EXPERIMENTS.items():
            results = self.load_experiment_results(config.name)
            if results:
                self.results[config.name] = results
                logger.info(f"Loaded: {config.name}")
    
    def generate_comparison_table(self) -> str:
        """Generate ASCII table comparing all experiments"""
        if not self.results:
            return "No results loaded"
        
        # Collect data
        rows = []
        headers = ["Experiment", "MRR", "NDCG", "Retrieval Time (ms)"]
        
        # Check if any experiment has generation metrics
        has_generation = any(r.get('generation') for r in self.results.values())
        if has_generation:
            headers.extend(["BLEU", "ROUGE-L", "BERTScore"])
        
        # Check if any experiment has hallucination metrics
        has_hallucination = any(r.get('hallucination') for r in self.results.values())
        if has_hallucination:
            headers.extend(["Hallucination Rate", "Faithfulness"])
        
        headers.append("Queries")
        
        # Build rows
        for exp_name, results in sorted(self.results.items()):
            row = [
                exp_name.replace('_', ' ').title(),
                f"{results['retrieval']['mrr']:.3f}",
                f"{results['retrieval']['ndcg']:.3f}",
                f"{results['retrieval']['mean_retrieval_time']:.1f}",
            ]
            
            if has_generation and results.get('generation'):
                gen = results['generation']
                row.extend([
                    f"{gen.get('bleu_score', 0):.3f}",
                    f"{gen.get('rouge_l', 0):.3f}",
                    f"{gen.get('bertscore_f1', 0):.3f}",
                ])
            elif has_generation:
                row.extend(["—", "—", "—"])
            
            if has_hallucination and results.get('hallucination'):
                hall = results['hallucination']
                row.extend([
                    f"{hall.get('hallucination_rate', 0):.1%}",
                    f"{hall.get('faithfulness_score', 0):.3f}",
                ])
            elif has_hallucination:
                row.extend(["—", "—"])
            
            row.append(str(results.get('num_queries', 0)))
            rows.append(row)
        
        # Build ASCII table
        table = self._build_ascii_table(headers, rows)
        return table
    
    @staticmethod
    def _build_ascii_table(headers: List[str], rows: List[List[str]]) -> str:
        """Build ASCII table from headers and rows"""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build table
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_row = "|" + "|".join(
            f" {h:{w}} " for h, w in zip(headers, col_widths)
        ) + "|"
        
        lines = [separator, header_row, separator]
        
        for row in rows:
            data_row = "|" + "|".join(
                f" {str(cell):{w}} " for cell, w in zip(row, col_widths)
            ) + "|"
            lines.append(data_row)
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def generate_csv_report(self, output_path: Path) -> Path:
        """Generate CSV file with all results"""
        if not self.results:
            logger.warning("No results to generate report")
            return output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all unique metrics
        all_keys = set()
        for results in self.results.values():
            all_keys.update(results.keys())
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['experiment'] + sorted(all_keys))
            writer.writeheader()
            
            for exp_name, results in sorted(self.results.items()):
                row = {'experiment': exp_name}
                row.update(results)
                writer.writerow(row)
        
        logger.info(f"Saved CSV report: {output_path}")
        return output_path
    
    def generate_markdown_report(self, output_path: Path) -> Path:
        """Generate Markdown report with results"""
        if not self.results:
            logger.warning("No results to generate report")
            return output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""# Ablation Study Results Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report compares performance of 5 different RAG system configurations:

1. **Baseline RAG** - Dense + Sparse retrieval
2. **Baseline + Embedding Tuning** - Fine-tuned embeddings
3. **Baseline + Reranker** - Cross-encoder reranker
4. **Baseline + LLM Fine-tuning** - Fine-tuned answer generation
5. **Fully Optimized** - All components enabled

---

## Results Comparison

### Retrieval Metrics

"""
        
        # Add retrieval metrics table
        content += "```\n"
        content += self.generate_comparison_table()
        content += "\n```\n\n"
        
        # Add detailed results
        content += "## Detailed Results\n\n"
        
        for exp_name, results in sorted(self.results.items()):
            content += f"### {exp_name.replace('_', ' ').title()}\n\n"
            content += f"**Timestamp:** {results.get('timestamp', 'N/A')}\n"
            content += f"**Queries:** {results.get('num_queries', 0)}\n"
            content += f"**Total Time:** {results.get('total_time_seconds', 0):.1f}s\n\n"
            
            if results.get('retrieval'):
                ret = results['retrieval']
                content += "**Retrieval Metrics:**\n"
                content += f"- MRR: {ret.get('mrr', 0):.3f}\n"
                content += f"- NDCG: {ret.get('ndcg', 0):.3f}\n"
                content += f"- Avg Retrieval Time: {ret.get('mean_retrieval_time', 0):.1f}ms\n"
                if ret.get('recall_at_k'):
                    content += f"- Recall@5: {ret['recall_at_k'].get(5, 0):.3f}\n"
                    content += f"- Recall@10: {ret['recall_at_k'].get(10, 0):.3f}\n"
                content += "\n"
            
            if results.get('generation'):
                gen = results['generation']
                content += "**Generation Metrics:**\n"
                content += f"- BLEU: {gen.get('bleu_score', 0):.3f}\n"
                content += f"- ROUGE-L: {gen.get('rouge_l', 0):.3f}\n"
                content += f"- BERTScore F1: {gen.get('bertscore_f1', 0):.3f}\n"
                content += "\n"
            
            if results.get('hallucination'):
                hall = results['hallucination']
                content += "**Hallucination Metrics:**\n"
                content += f"- Hallucination Rate: {hall.get('hallucination_rate', 0):.1%}\n"
                content += f"- Faithfulness Score: {hall.get('faithfulness_score', 0):.3f}\n"
                content += "\n"
            
            if results.get('notes'):
                content += f"**Notes:** {results['notes']}\n\n"
        
        # Add conclusions
        content += """---

## Key Findings

### Best Configuration by Metric

"""
        
        # Find best for each metric
        if self.results:
            best_mrr = max(self.results.items(), 
                          key=lambda x: x[1].get('retrieval', {}).get('mrr', 0))
            content += f"- **Best MRR:** {best_mrr[0]} ({best_mrr[1]['retrieval']['mrr']:.3f})\n"
            
            if any(r.get('generation') for r in self.results.values()):
                best_bleu = max(
                    ((k, v) for k, v in self.results.items() if v.get('generation')),
                    key=lambda x: x[1]['generation'].get('bleu_score', 0),
                    default=(None, {})
                )
                if best_bleu[0]:
                    content += f"- **Best BLEU:** {best_bleu[0]} ({best_bleu[1]['generation']['bleu_score']:.3f})\n"
        
        content += f"""

## Recommendations

1. **Embedding Tuning Impact:** Compare experiment 2 vs 1 to see embedding benefit
2. **Reranking Impact:** Compare experiment 3 vs 1 to see reranker benefit
3. **LLM Fine-tuning Impact:** Compare experiment 4 vs 1 to see LLM benefit
4. **Combined Effects:** Compare experiment 5 vs 1 to see total system improvement

---

*Report generated by PROMPT 12 Ablation Study Framework*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved Markdown report: {output_path}")
        return output_path
    
    def print_summary(self) -> None:
        """Print results summary to console"""
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("=" * 80 + "\n")
        print(self.generate_comparison_table())
        print("\n" + "=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    experiments_dir = Path("results")
    
    comparator = ResultsComparator(experiments_dir)
    comparator.load_all_results()
    comparator.print_summary()
    
    # Generate reports
    comparator.generate_csv_report(experiments_dir / "comparison_results.csv")
    comparator.generate_markdown_report(experiments_dir / "ABLATION_RESULTS.md")
