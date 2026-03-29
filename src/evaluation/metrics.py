#!/usr/bin/env python3
"""
PROMPT 11: Evaluation Metrics - Retrieval & QA Metrics
Recall@k, MRR, nDCG, Exact Match, F1, BLEU, ROUGE
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics"""
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_5: float
    ndcg_at_10: float
    mean_rank: float

class RetrievalEvaluator:
    """Evaluate retrieval quality"""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """
        Recall@k: What fraction of relevant docs are in top-k?
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevant_ids: List of relevant document IDs (gold standard)
            k: Cutoff
            
        Returns:
            Recall@k score (0-1)
        """
        if len(relevant_ids) == 0:
            return 1.0  # Edge case: no relevant docs
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        num_relevant_retrieved = len(retrieved_at_k & relevant_set)
        recall = num_relevant_retrieved / len(relevant_set)
        
        return min(recall, 1.0)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        MRR: Average of 1/rank of first relevant doc
        
        Args:
            retrieved_ids: Retrieved docs (ordered)
            relevant_ids: Gold standard relevant docs
            
        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0  # No relevant doc found
    
    @staticmethod
    def ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """
        nDCG@k: Normalized Discounted Cumulative Gain
        Accounts for ranking quality
        
        Args:
            retrieved_ids: Retrieved docs (ordered)
            relevant_ids: Gold standard relevant docs
            k: Cutoff
            
        Returns:
            nDCG@k score (0-1)
        """
        relevant_set = set(relevant_ids)
        
        # Build relevance scores (1 if relevant, 0 otherwise)
        relevances = [1.0 if doc_id in relevant_set else 0.0 
                      for doc_id in retrieved_ids[:k]]
        
        # DCG = sum(rel_i / log2(i+1))
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
        
        # Ideal DCG: perfect ranking (all relevant docs first)
        ideal_relevances = [1.0] * min(len(relevant_ids), k) + \
                          [0.0] * max(0, k - len(relevant_ids))
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        if idcg == 0:
            return 1.0 if dcg == 0 else 0.0
        
        return dcg / idcg
    
    @staticmethod
    def evaluate(retrieved_ids: List[str], relevant_ids: List[str]) -> RetrievalMetrics:
        """
        Compute all retrieval metrics
        
        Args:
            retrieved_ids: Retrieved docs (ordered by score)
            relevant_ids: Gold standard relevant docs
            
        Returns:
            RetrievalMetrics object
        """
        recall_5 = RetrievalEvaluator.recall_at_k(retrieved_ids, relevant_ids, k=5)
        recall_10 = RetrievalEvaluator.recall_at_k(retrieved_ids, relevant_ids, k=10)
        mrr = RetrievalEvaluator.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg_5 = RetrievalEvaluator.ndcg(retrieved_ids, relevant_ids, k=5)
        ndcg_10 = RetrievalEvaluator.ndcg(retrieved_ids, relevant_ids, k=10)
        
        # Mean rank of first relevant doc
        relevant_set = set(relevant_ids)
        first_relevant_rank = next(
            (i + 1 for i, doc_id in enumerate(retrieved_ids) if doc_id in relevant_set),
            len(retrieved_ids) + 1
        )
        
        return RetrievalMetrics(
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            mrr=mrr,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            mean_rank=float(first_relevant_rank)
        )


@dataclass
class QAMetrics:
    """QA answer evaluation metrics"""
    exact_match: float  # Binary: answer matches exactly
    token_f1: float  # Token-level F1 score
    bleu: float  # BLEU score (precision of n-grams)
    rouge_l: float  # ROUGE-L (longest common subsequence)
    

class QAEvaluator:
    """Evaluate answer quality"""
    
    @staticmethod
    def exact_match(predicted: str, gold: str) -> float:
        """Exact Match: 1 if prediction == gold, 0 otherwise"""
        pred_norm = predicted.strip().lower()
        gold_norm = gold.strip().lower()
        return 1.0 if pred_norm == gold_norm else 0.0
    
    @staticmethod
    def token_f1(predicted: str, gold: str) -> float:
        """
        Token-level F1: overlap of tokens
        
        Args:
            predicted: Predicted answer
            gold: Gold standard answer
            
        Returns:
            F1 score (0-1)
        """
        # Tokenize by whitespace
        pred_tokens = set(predicted.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if len(gold_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        # Precision & Recall
        common = pred_tokens & gold_tokens
        
        if len(pred_tokens) == 0:
            precision = 0.0
        else:
            precision = len(common) / len(pred_tokens)
        
        recall = len(common) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def bleu(predicted: str, gold: str, max_n: int = 4) -> float:
        """
        BLEU: N-gram precision (cumulative)
        
        Args:
            predicted: Predicted answer
            gold: Gold standard answer
            max_n: Maximum n-gram size
            
        Returns:
            BLEU score (0-1)
        """
        pred_tokens = predicted.lower().split()
        gold_tokens = gold.lower().split()
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0 if len(pred_tokens) != len(gold_tokens) else 1.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
            pred_ngrams = QAEvaluator._get_ngrams(pred_tokens, n)
            gold_ngrams = QAEvaluator._get_ngrams(gold_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = sum(min(pred_ngrams[ng], gold_ngrams.get(ng, 0)) 
                         for ng in pred_ngrams)
            precision = matches / len(pred_ngrams)
            precisions.append(precision)
        
        if any(p == 0 for p in precisions):
            return 0.0
        
        # Brevity penalty
        brevity_penalty = min(1.0, len(pred_tokens) / max(len(gold_tokens), 1))
        
        # Geometric mean of precisions
        geo_mean = np.exp(np.mean(np.log(precisions)))
        bleu = brevity_penalty * geo_mean
        
        return bleu
    
    @staticmethod
    def rouge_l(predicted: str, gold: str) -> float:
        """
        ROUGE-L: F1 of Longest Common Subsequence
        
        Args:
            predicted: Predicted answer
            gold: Gold standard answer
            
        Returns:
            ROUGE-L F1 score (0-1)
        """
        pred_tokens = predicted.lower().split()
        gold_tokens = gold.lower().split()
        
        if len(gold_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        lcs_length = QAEvaluator._lcs_length(pred_tokens, gold_tokens)
        
        # Precision & Recall based on LCS
        if len(pred_tokens) == 0:
            precision = 0.0
        else:
            precision = lcs_length / len(pred_tokens)
        
        recall = lcs_length / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
        """Get n-gram counts"""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            ngrams[ng] += 1
        return ngrams
    
    @staticmethod
    def evaluate(predicted: str, gold: str) -> QAMetrics:
        """
        Compute all QA metrics
        
        Args:
            predicted: Predicted answer
            gold: Gold standard answer
            
        Returns:
            QAMetrics object
        """
        em = QAEvaluator.exact_match(predicted, gold)
        f1 = QAEvaluator.token_f1(predicted, gold)
        bleu = QAEvaluator.bleu(predicted, gold)
        rouge_l = QAEvaluator.rouge_l(predicted, gold)
        
        return QAMetrics(
            exact_match=em,
            token_f1=f1,
            bleu=bleu,
            rouge_l=rouge_l
        )
