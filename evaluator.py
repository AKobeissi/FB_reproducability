"""
Evaluation Module for RAG Systems
Includes statistical metrics (BLEU, ROUGE), semantic metrics (BERTScore), 
and LLM-as-judge evaluation

Uses standard packages:
- sacrebleu for BLEU scores
- rouge-score for ROUGE metrics
- bert-score for semantic similarity
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter
import re

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluation suite for RAG systems"""
    
    def __init__(self, use_bertscore: bool = True, use_llm_judge: bool = True):
        """
        Initialize evaluator with specified metrics
        
        Args:
            use_bertscore: Whether to use BERTScore (requires transformers)
            use_llm_judge: Whether to use LLM as judge
        """
        self.use_bertscore = use_bertscore
        self.use_llm_judge = use_llm_judge
        
        # Initialize BLEU scorer (sacrebleu)
        self.bleu_scorer = None
        try:
            import sacrebleu
            self.bleu_scorer = sacrebleu
            logger.info("SacreBLEU initialized successfully")
        except ImportError:
            logger.warning("SacreBLEU not available. Install with: pip install sacrebleu")
            self.bleu_scorer = None
        
        # Initialize ROUGE scorer
        self.rouge_scorer = None
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            logger.info("ROUGE scorer initialized successfully")
        except ImportError:
            logger.warning("rouge-score not available. Install with: pip install rouge-score")
            self.rouge_scorer = None
        
        # Initialize BERTScore
        self.bertscore_scorer = None
        if use_bertscore:
            try:
                from bert_score import BERTScorer
                self.bertscore_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                logger.info("BERTScore initialized successfully")
            except ImportError:
                logger.warning("BERTScore not available. Install with: pip install bert-score")
                self.use_bertscore = False
        
        logger.info(f"Evaluator initialized - BLEU: {self.bleu_scorer is not None}, "
                   f"ROUGE: {self.rouge_scorer is not None}, "
                   f"BERTScore: {self.use_bertscore}, LLM Judge: {self.use_llm_judge}")
    
    # ========== Statistical Metrics ==========
    
    def compute_bleu(self, prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
        """
        Compute BLEU scores using SacreBLEU (BLEU-1 through BLEU-4)
        
        Args:
            prediction: Generated text
            reference: Reference text
            max_n: Maximum n-gram size
            
        Returns:
            Dictionary with BLEU scores
        """
        # If sacrebleu is unavailable, use fallback implementation
        if self.bleu_scorer is None:
            logger.warning("SacreBLEU not available, using fallback BLEU")
            return self._compute_bleu_fallback(prediction, reference, max_n)

        # Try using sacrebleu's sentence_bleu for an overall BLEU, but compute
        # per-n BLEU using the stable fallback implementation to avoid
        # depending on advanced sacrebleu internal params.
        try:
            overall = self.bleu_scorer.sentence_bleu(prediction, [reference]).score / 100.0
        except Exception as e:
            logger.warning(f"SacreBLEU sentence_bleu failed ({e}), using fallback")
            return self._compute_bleu_fallback(prediction, reference, max_n)

        # Compute BLEU-1..BLEU-N via the fallback implementation for stability
        n_scores = self._compute_bleu_fallback(prediction, reference, max_n)
        # Ensure bleu_4 reflects sacrebleu overall BLEU if available
        if 'bleu_4' in n_scores:
            n_scores['bleu_4'] = overall

        logger.debug(f"BLEU scores: {n_scores}")
        return n_scores
    
    def _compute_bleu_fallback(self, prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
        """Fallback BLEU implementation if sacrebleu not available"""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        scores = {}
        for n in range(1, max_n + 1):
            score = self._compute_bleu_n(pred_tokens, ref_tokens, n)
            scores[f'bleu_{n}'] = score
        
        return scores
    
    def _compute_bleu_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Compute BLEU score for specific n-gram size"""
        if len(pred_tokens) < n:
            return 0.0
        
        # Generate n-grams
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        
        if not pred_ngrams or not ref_ngrams:
            return 0.0
        
        # Count matches
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        matches = sum((pred_counter & ref_counter).values())
        total = len(pred_ngrams)
        
        # Compute precision with brevity penalty
        precision = matches / total if total > 0 else 0.0
        
        # Brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        return bp * precision
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L scores using rouge-score package
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            Dictionary with ROUGE scores (precision, recall, F1 for each type)
        """
        if self.rouge_scorer is None:
            logger.warning("rouge-score package not available, falling back to custom implementation")
            return self._compute_rouge_fallback(prediction, reference)
        
        scores = {}
        
        # Compute ROUGE scores using the package
        rouge_scores = self.rouge_scorer.score(reference, prediction)
        
        # Extract ROUGE-1 scores
        scores['rouge_1_precision'] = rouge_scores['rouge1'].precision
        scores['rouge_1_recall'] = rouge_scores['rouge1'].recall
        scores['rouge_1_f1'] = rouge_scores['rouge1'].fmeasure
        
        # Extract ROUGE-2 scores
        scores['rouge_2_precision'] = rouge_scores['rouge2'].precision
        scores['rouge_2_recall'] = rouge_scores['rouge2'].recall
        scores['rouge_2_f1'] = rouge_scores['rouge2'].fmeasure
        
        # Extract ROUGE-L scores
        scores['rouge_l_precision'] = rouge_scores['rougeL'].precision
        scores['rouge_l_recall'] = rouge_scores['rougeL'].recall
        scores['rouge_l_f1'] = rouge_scores['rougeL'].fmeasure
        
        logger.debug(f"ROUGE scores: {scores}")
        return scores
    
    def _compute_rouge_fallback(self, prediction: str, reference: str) -> Dict[str, float]:
        """Fallback ROUGE implementation if rouge-score not available"""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        scores = {}
        
        # ROUGE-1
        rouge_1 = self._compute_rouge_n(pred_tokens, ref_tokens, 1)
        scores.update({f'rouge_1_{k}': v for k, v in rouge_1.items()})
        
        # ROUGE-2
        rouge_2 = self._compute_rouge_n(pred_tokens, ref_tokens, 2)
        scores.update({f'rouge_2_{k}': v for k, v in rouge_2.items()})
        
        # ROUGE-L (Longest Common Subsequence)
        rouge_l = self._compute_rouge_l(pred_tokens, ref_tokens)
        scores.update({f'rouge_l_{k}': v for k, v in rouge_l.items()})
        
        return scores
    
    def _compute_rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
        """Compute ROUGE-N scores"""
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        
        matches = sum((pred_ngrams & ref_ngrams).values())
        
        precision = matches / sum(pred_ngrams.values()) if pred_ngrams else 0.0
        recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _compute_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
        """Compute ROUGE-L based on longest common subsequence"""
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # ========== Semantic Metrics ==========
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Compute BERTScore for semantic similarity
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not self.use_bertscore or self.bertscore_scorer is None:
            logger.warning("BERTScore not available")
            return {'precision': [], 'recall': [], 'f1': []}
        
        try:
            P, R, F1 = self.bertscore_scorer.score(predictions, references)
            
            scores = {
                'precision': P.tolist(),
                'recall': R.tolist(),
                'f1': F1.tolist(),
                'precision_mean': P.mean().item(),
                'recall_mean': R.mean().item(),
                'f1_mean': F1.mean().item()
            }
            
            logger.info(f"BERTScore - P: {scores['precision_mean']:.4f}, R: {scores['recall_mean']:.4f}, F1: {scores['f1_mean']:.4f}")
            return scores
            
        except Exception as e:
            logger.error(f"Error computing BERTScore: {str(e)}")
            return {'precision': [], 'recall': [], 'f1': []}
    
    # ========== Retrieval Metrics ==========
    
    def compute_retrieval_metrics(self, retrieved_docs: List[str], 
                                  gold_evidence: str,
                                  top_k: int = None) -> Dict[str, float]:
        """
        Compute retrieval metrics comparing retrieved documents to gold evidence
        
        Args:
            retrieved_docs: List of retrieved document texts
            gold_evidence: Gold standard evidence text
            top_k: Consider only top k documents (None for all)
            
        Returns:
            Dictionary with retrieval metrics
        """
        if top_k is not None:
            retrieved_docs = retrieved_docs[:top_k]
        
        # Check if gold evidence appears in retrieved docs (exact match)
        exact_match = any(gold_evidence.strip() in doc for doc in retrieved_docs)
        
        # Check for partial matches (token overlap)
        gold_tokens = set(self._tokenize(gold_evidence))
        
        overlaps = []
        for doc in retrieved_docs:
            doc_tokens = set(self._tokenize(doc))
            if len(gold_tokens) > 0:
                overlap = len(gold_tokens & doc_tokens) / len(gold_tokens)
                overlaps.append(overlap)
            else:
                overlaps.append(0.0)
        
        metrics = {
            'exact_match': 1.0 if exact_match else 0.0,
            'max_token_overlap': max(overlaps) if overlaps else 0.0,
            'mean_token_overlap': np.mean(overlaps) if overlaps else 0.0,
            'num_retrieved': len(retrieved_docs)
        }
        
        logger.debug(f"Retrieval metrics: {metrics}")
        return metrics
    
    # ========== LLM as Judge ==========
    
    def llm_judge_correctness(self, question: str, prediction: str, 
                            reference: str, model: str = "gpt-4") -> Dict[str, Any]:
        """
        Use LLM as judge to evaluate correctness
        
        Args:
            question: Original question
            prediction: Generated answer
            reference: Reference answer
            model: Model to use for judgment
            
        Returns:
            Dictionary with judgment (correct/incorrect) and explanation
        """
        if not self.use_llm_judge:
            logger.warning("LLM judge not enabled")
            return {'correct': None, 'explanation': 'LLM judge not enabled', 'confidence': None}
        
        try:
            # Create evaluation prompt
            prompt = self._create_judge_prompt(question, prediction, reference)
            
            # Call LLM (placeholder - implement with actual API)
            # This is a template - you'll need to implement with your LLM API
            judgment = self._call_llm_judge(prompt, model)
            
            logger.info(f"LLM Judge: {judgment['correct']} (confidence: {judgment.get('confidence', 'N/A')})")
            return judgment
            
        except Exception as e:
            logger.error(f"Error in LLM judge: {str(e)}")
            return {'correct': None, 'explanation': str(e), 'confidence': None}
    
    def _create_judge_prompt(self, question: str, prediction: str, reference: str) -> str:
        """Create prompt for LLM judge"""
        return f"""You are evaluating the correctness of an answer to a financial question.

Question: {question}

Reference Answer: {reference}

Predicted Answer: {prediction}

Task: Determine if the predicted answer is correct compared to the reference answer.
Consider the answer correct if:
1. The main facts and figures match
2. The core meaning is preserved
3. Minor wording differences are acceptable

Respond in this exact format:
CORRECT: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Brief explanation of your judgment]
"""
    
    def _call_llm_judge(self, prompt: str, model: str) -> Dict[str, Any]:
        """
        Call LLM API for judgment using HuggingFace pipeline
        """
        if not hasattr(self, '_judge_pipeline') or self._judge_pipeline is None:
            logger.warning("LLM judge pipeline not initialized")
            return {
                'correct': None,
                'confidence': None,
                'explanation': 'LLM judge pipeline not initialized'
            }
        
        try:
            # Generate judgment
            response = self._judge_pipeline(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                pad_token_id=self._judge_pipeline.tokenizer.eos_token_id
            )
            
            # Extract generated text
            judgment_text = response[0]['generated_text'].strip()
            
            # Parse the judgment
            correct = None
            confidence = None
            explanation = judgment_text
            
            # Try to extract structured response
            if "CORRECT:" in judgment_text:
                correct_line = [line for line in judgment_text.split('\n') if 'CORRECT:' in line]
                if correct_line:
                    correct_val = correct_line[0].split('CORRECT:')[1].strip().upper()
                    correct = 'YES' in correct_val or 'TRUE' in correct_val
            
            if "CONFIDENCE:" in judgment_text:
                conf_line = [line for line in judgment_text.split('\n') if 'CONFIDENCE:' in line]
                if conf_line:
                    confidence = conf_line[0].split('CONFIDENCE:')[1].strip().upper()
            
            if "EXPLANATION:" in judgment_text:
                expl_lines = judgment_text.split('EXPLANATION:')
                if len(expl_lines) > 1:
                    explanation = expl_lines[1].strip()
            
            logger.info(f"LLM Judge - Correct: {correct}, Confidence: {confidence}")
            
            return {
                'correct': correct,
                'confidence': confidence,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Error in LLM judge: {str(e)}")
            return {
                'correct': None,
                'confidence': None,
                'explanation': f'Error: {str(e)}'
            }
    
    # ========== Utility Methods ==========
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def evaluate_generation(self, prediction: str, reference: str, 
                          question: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated text
        
        Args:
            prediction: Generated text
            reference: Reference text
            question: Original question (optional, for LLM judge)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("GENERATION EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Prediction: {prediction[:200]}...")
        logger.info(f"Reference: {reference[:200]}...")
        
        results = {}
        
        # Statistical metrics
        results['bleu'] = self.compute_bleu(prediction, reference)
        results['rouge'] = self.compute_rouge(prediction, reference)
        
        # Semantic metrics
        if self.use_bertscore:
            bert_scores = self.compute_bertscore([prediction], [reference])
            results['bertscore'] = {
                'precision': bert_scores['precision'][0] if bert_scores['precision'] else None,
                'recall': bert_scores['recall'][0] if bert_scores['recall'] else None,
                'f1': bert_scores['f1'][0] if bert_scores['f1'] else None
            }
        
        # LLM judge
        if self.use_llm_judge and question:
            results['llm_judge'] = self.llm_judge_correctness(question, prediction, reference)
        
        # Log summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"  BLEU-4: {results['bleu'].get('bleu_4', 0):.4f}")
        logger.info(f"  ROUGE-L F1: {results['rouge'].get('rouge_l_f1', 0):.4f}")
        if 'bertscore' in results and results['bertscore']['f1']:
            logger.info(f"  BERTScore F1: {results['bertscore']['f1']:.4f}")
        
        logger.info("=" * 80)
        
        return results
    
    def evaluate_batch(self, predictions: List[str], references: List[str],
                      questions: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            questions: List of questions (optional)
            
        Returns:
            Dictionary with aggregated metrics
        """
        logger.info(f"Evaluating batch of {len(predictions)} predictions")
        
        all_results = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            question = questions[i] if questions else None
            result = self.evaluate_generation(pred, ref, question)
            all_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        logger.info("\nBatch Evaluation Summary:")
        logger.info(f"  Mean BLEU-4: {aggregated['bleu_4_mean']:.4f}")
        logger.info(f"  Mean ROUGE-L F1: {aggregated['rouge_l_f1_mean']:.4f}")
        if 'bertscore_f1_mean' in aggregated:
            logger.info(f"  Mean BERTScore F1: {aggregated['bertscore_f1_mean']:.4f}")
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate results from multiple evaluations"""
        aggregated = {}
        
        # Aggregate BLEU scores
        for n in range(1, 5):
            key = f'bleu_{n}'
            scores = [r['bleu'][key] for r in results if 'bleu' in r and key in r['bleu']]
            if scores:
                aggregated[f'{key}_mean'] = np.mean(scores)
                aggregated[f'{key}_std'] = np.std(scores)
        
        # Aggregate ROUGE scores
        rouge_metrics = ['rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1']
        for metric in rouge_metrics:
            scores = [r['rouge'][metric] for r in results if 'rouge' in r and metric in r['rouge']]
            if scores:
                aggregated[f'{metric}_mean'] = np.mean(scores)
                aggregated[f'{metric}_std'] = np.std(scores)
        
        # Aggregate BERTScore
        if any('bertscore' in r for r in results):
            for metric in ['precision', 'recall', 'f1']:
                scores = [r['bertscore'][metric] for r in results 
                         if 'bertscore' in r and r['bertscore'][metric] is not None]
                if scores:
                    aggregated[f'bertscore_{metric}_mean'] = np.mean(scores)
                    aggregated[f'bertscore_{metric}_std'] = np.std(scores)
        
        return aggregated


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the evaluator
    evaluator = Evaluator(use_bertscore=False, use_llm_judge=False)
    
    # Test data
    prediction = "The company's revenue increased by 15% to $10 million in Q4."
    reference = "Revenue grew 15 percent reaching $10M in the fourth quarter."
    
    # Evaluate
    results = evaluator.evaluate_generation(prediction, reference)
    
    print("\n" + "="*80)
    print("EVALUATOR TEST COMPLETE")
    print("="*80)