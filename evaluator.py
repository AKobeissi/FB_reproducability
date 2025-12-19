"""
Evaluation Module for RAG Systems
Includes statistical metrics (BLEU, ROUGE), semantic metrics (BERTScore),
RAGAS-style holistic evaluations, retrieval diagnostics, and HuggingFace-based
LLM-as-judge scoring using the LlamaIndex CorrectnessEvaluator prompt.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

try:
    from llama_index.core.evaluation.correctness import (
        DEFAULT_SYSTEM_TEMPLATE as LLAMAINDEX_SYSTEM_PROMPT,
        DEFAULT_USER_TEMPLATE as LLAMAINDEX_USER_PROMPT,
    )
    _HAS_LLAMA_INDEX = True
except Exception:  # pragma: no cover
    LLAMAINDEX_SYSTEM_PROMPT = None
    LLAMAINDEX_USER_PROMPT = None
    _HAS_LLAMA_INDEX = False

try:  # pragma: no cover - optional dependency
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    _HAS_RAGAS = True
except Exception:
    ragas_evaluate = None
    EvaluationDataset = None
    RagasHuggingFaceEmbeddings = None
    LangchainLLMWrapper = None
    faithfulness = answer_relevancy = context_precision = context_recall = None
    _HAS_RAGAS = False

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluation suite for RAG systems"""

    def __init__(
        self,
        use_bertscore: bool = True,
        use_llm_judge: bool = True,
        use_ragas: bool = True,
        ragas_embedding_model: Optional[str] = None,
        ragas_device: Optional[str] = None,
    ):
        """
        Initialize evaluator with specified metrics

        Args:
            use_bertscore: Whether to use BERTScore (requires transformers)
            use_llm_judge: Whether to use LLM as judge
            use_ragas: Whether to run RAGAS-style holistic metrics
            ragas_embedding_model: Optional embedding model override for RAGAS
            ragas_device: Optional device hint (cpu/cuda) for RAGAS embeddings
        """
        self.use_bertscore = use_bertscore
        self.use_llm_judge = use_llm_judge
        self.use_ragas = use_ragas and _HAS_RAGAS

        self._ragas_embedding_model_name = ragas_embedding_model
        self._ragas_device = ragas_device
        self._ragas_embeddings = None
        self._ragas_metrics = (
            [faithfulness, answer_relevancy, context_precision, context_recall]
            if self.use_ragas and all(
                metric is not None
                for metric in [faithfulness, answer_relevancy, context_precision, context_recall]
            )
            else []
        )
        self._ragas_llm_cache: Dict[str, Any] = {}
        self._judge_pipeline = None
        self._judge_max_new_tokens = 200
        if use_ragas and not _HAS_RAGAS:
            logger.warning(
                "RAGAS not available. Install 'ragas>=0.3.9' to enable holistic metrics."
            )

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
        
        logger.info(
            "Evaluator initialized - BLEU: %s, ROUGE: %s, BERTScore: %s, "
            "LLM Judge: %s, RAGAS: %s",
            self.bleu_scorer is not None,
            self.rouge_scorer is not None,
            self.use_bertscore,
            self.use_llm_judge,
            self.use_ragas,
        )
    
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

    # ========== RAGAS Metrics ==========

    def configure_ragas(self, embedding_model: Optional[str] = None, device: Optional[str] = None):
        """Configure default embedding model/device for RAGAS evaluations."""
        if embedding_model:
            self._ragas_embedding_model_name = embedding_model
            self._ragas_embeddings = None  # reset cache
        if device:
            self._ragas_device = device
            self._ragas_embeddings = None

    def _get_ragas_embeddings(self):
        if not self.use_ragas or RagasHuggingFaceEmbeddings is None:
            return None
        if self._ragas_embeddings is not None:
            return self._ragas_embeddings

        model_name = self._ragas_embedding_model_name or "sentence-transformers/all-mpnet-base-v2"
        try:
            self._ragas_embeddings = RagasHuggingFaceEmbeddings(
                model=model_name,
                device=self._ragas_device,
                normalize_embeddings=True,
            )
            logger.info("Initialized RAGAS embeddings (%s)", model_name)
        except Exception as exc:
            logger.warning("Failed to initialize RAGAS embeddings (%s): %s", model_name, exc)
            self._ragas_embeddings = None
        return self._ragas_embeddings

    def _get_ragas_llm(self, langchain_llm: Any):
        if not self.use_ragas or LangchainLLMWrapper is None:
            return None
        if langchain_llm is None:
            logger.debug("RAGAS requires a LangChain LLM wrapper; none provided.")
            return None
        cache_key = id(langchain_llm)
        cached = self._ragas_llm_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            wrapper = LangchainLLMWrapper(langchain_llm)
            self._ragas_llm_cache[cache_key] = wrapper
            return wrapper
        except Exception as exc:
            logger.warning("Failed to wrap LangChain LLM for RAGAS: %s", exc)
            return None

    def compute_ragas_scores(
        self,
        question: str,
        prediction: str,
        contexts: List[str],
        reference_answer: str,
        reference_contexts: List[str],
        langchain_llm: Any = None,
    ) -> Dict[str, float]:
        """Compute RAGAS-style holistic metrics for a single sample."""
        if (
            not self.use_ragas
            or not self._ragas_metrics
            or ragas_evaluate is None
            or EvaluationDataset is None
        ):
            return {}

        ragas_llm = self._get_ragas_llm(langchain_llm)
        ragas_embeddings = self._get_ragas_embeddings()
        if ragas_llm is None or ragas_embeddings is None:
            return {}

        sample = {
            "user_input": question or "",
            "response": prediction or "",
            "retrieved_contexts": contexts or [],
            "reference": reference_answer or "",
            "reference_contexts": reference_contexts or [],
        }
        try:
            dataset = EvaluationDataset.from_list([sample])
            evaluation = ragas_evaluate(
                dataset,
                metrics=self._ragas_metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                show_progress=False,
            )
            ragas_scores: Dict[str, float] = {}
            for metric in self._ragas_metrics:
                name = getattr(metric, "name", None)
                if not name:
                    continue
                metric_values = evaluation[name]
                if metric_values:
                    ragas_scores[name] = float(metric_values[0])
            return ragas_scores
        except Exception as exc:
            logger.warning("RAGAS evaluation error: %s", exc)
            return {}
    
    # ========== Retrieval Metrics ==========
    
    def compute_retrieval_metrics(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        gold_segments: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        match_threshold: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Compute retrieval diagnostics (Recall@k / Hit@k) at document, page, and chunk level.

        Args:
            retrieved_chunks: List of chunk dictionaries with `text` and `metadata`.
            gold_segments: Structured gold evidence entries (each dict contains text/doc/page).
            top_k: Optional limit on evaluated retrievals.
            match_threshold: Token-overlap threshold required for chunk matches.
        """
        evaluated_chunks = retrieved_chunks[:top_k] if top_k else list(retrieved_chunks)
        gold_segments = gold_segments or []

        gold_texts = [seg.get("text", "") for seg in gold_segments if seg.get("text")]
        gold_concat = "\n\n".join(gold_texts)
        gold_docs = {seg.get("doc_name") for seg in gold_segments if seg.get("doc_name")}
        gold_pages = {
            (seg.get("doc_name"), seg.get("page"))
            for seg in gold_segments
            if seg.get("doc_name") is not None and seg.get("page") is not None
        }

        overlaps: List[float] = []
        exact_match = False

        doc_hits: Dict[str, int] = {}
        page_hits: Dict[Tuple[Any, Any], int] = {}
        matched_chunk_indices: set = set()
        chunk_hit_rank = None

        for idx, chunk in enumerate(evaluated_chunks):
            text = chunk.get("text", "") or ""
            metadata = chunk.get("metadata", {}) or {}
            doc_name = metadata.get("doc_name")
            page = metadata.get("page")
            chunk_tokens_overlap = (
                self._token_overlap_ratio(text, gold_concat) if gold_concat else 0.0
            )
            overlaps.append(chunk_tokens_overlap)

            if gold_concat and gold_concat.strip() and gold_concat.strip() in text:
                exact_match = True

            if doc_name in gold_docs and doc_name not in doc_hits:
                doc_hits[doc_name] = idx + 1

            page_key = (doc_name, page)
            if page_key in gold_pages and page_key not in page_hits:
                page_hits[page_key] = idx + 1

            for seg_idx, seg in enumerate(gold_segments):
                gold_text = seg.get("text") or ""
                if not gold_text:
                    continue
                overlap_score = self._token_overlap_ratio(text, gold_text)
                if overlap_score >= match_threshold:
                    matched_chunk_indices.add(seg_idx)
                    if chunk_hit_rank is None:
                        chunk_hit_rank = idx + 1
                    break

        doc_hit_rank = min(doc_hits.values()) if doc_hits else None
        page_hit_rank = min(page_hits.values()) if page_hits else None

        doc_hit = bool(doc_hits)
        page_hit = bool(page_hits)
        chunk_hit = bool(matched_chunk_indices)

        doc_recall = len(doc_hits) / len(gold_docs) if gold_docs else 0.0
        page_recall = len(page_hits) / len(gold_pages) if gold_pages else 0.0
        chunk_recall = (
            len(matched_chunk_indices) / len(gold_segments) if gold_segments else 0.0
        )

        if not evaluated_chunks:
            miss_reason = "no_retrievals"
        elif not doc_hit:
            miss_reason = "no_doc_match"
        elif doc_hit and not page_hit:
            miss_reason = "doc_match_page_miss"
        elif page_hit and not chunk_hit:
            miss_reason = "page_match_chunk_miss"
        else:
            miss_reason = "chunk_match"

        metrics = {
            "num_retrieved": len(evaluated_chunks),
            "exact_match": 1.0 if exact_match else 0.0,
            "max_token_overlap": max(overlaps) if overlaps else 0.0,
            "mean_token_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
            "doc_hit_at_k": doc_hit,
            "page_hit_at_k": page_hit,
            "chunk_hit_at_k": chunk_hit,
            "doc_recall_at_k": doc_recall,
            "page_recall_at_k": page_recall,
            "chunk_recall_at_k": chunk_recall,
            "doc_hit_rank": doc_hit_rank,
            "page_hit_rank": page_hit_rank,
            "chunk_hit_rank": chunk_hit_rank,
            "failure_reason": miss_reason,
        }

        logger.debug("Retrieval metrics: %s", metrics)
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
        """Create prompt for LLM judge using LlamaIndex correctness template when available."""
        system_prompt = (
            LLAMAINDEX_SYSTEM_PROMPT.strip()
            if _HAS_LLAMA_INDEX and LLAMAINDEX_SYSTEM_PROMPT
            else "You are an expert evaluation system for a question answering chatbot."
        )
        user_template = (
            LLAMAINDEX_USER_PROMPT
            if _HAS_LLAMA_INDEX and LLAMAINDEX_USER_PROMPT
            else "## User Query\n{query}\n\n## Reference Answer\n{reference_answer}\n\n## Generated Answer\n{generated_answer}"
        )
        user_message = user_template.format(
            query=question or "(missing question)",
            reference_answer=reference or "(no reference provided)",
            generated_answer=prediction or "(no answer provided)",
        )
        prompt_body = f"{system_prompt.strip()}\n\n{user_message.strip()}\n\nReturn the numeric score (1-5) on the first line and provide reasoning on the next line."
        return prompt_body
    
    def set_judge_pipeline(self, pipeline_obj: Any, max_new_tokens: Optional[int] = None):
        """
        Attach a HuggingFace text-generation pipeline (or compatible callable) that will
        score answers when `use_llm_judge` is True.

        Args:
            pipeline_obj: Typically a `transformers.pipeline("text-generation", ...)`.
            max_new_tokens: Optional override for the number of tokens to sample per judgment.
        """
        self._judge_pipeline = pipeline_obj
        if max_new_tokens is not None:
            self._judge_max_new_tokens = max_new_tokens
        logger.info(
            "LLM judge pipeline attached (%s tokens max).",
            self._judge_max_new_tokens,
        )
    
    def _call_llm_judge(self, prompt: str, model: str) -> Dict[str, Any]:
        """
        Call LLM API for judgment using HuggingFace pipeline
        """
        if self._judge_pipeline is None:
            logger.warning("LLM judge pipeline not initialized")
            return {
                'correct': None,
                'confidence': None,
                'explanation': 'LLM judge pipeline not initialized'
            }
        
        try:
            # Apply chat template if available
            prompt_formatted = prompt
            if hasattr(self._judge_pipeline, "tokenizer") and hasattr(self._judge_pipeline.tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    prompt_formatted = self._judge_pipeline.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback to Llama-style if template application fails
                    prompt_formatted = f"<s>[INST] {prompt} [/INST]"
            else:
                 # Fallback to Llama-style
                prompt_formatted = f"<s>[INST] {prompt} [/INST]"

            max_new_tokens = getattr(self, "_judge_max_new_tokens", 200)
            response = self._judge_pipeline(
                prompt_formatted,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=self._judge_pipeline.tokenizer.eos_token_id,
            )

            judgment_text = response[0]['generated_text']
            # Remove prompt from generated text
            if prompt_formatted and judgment_text.startswith(prompt_formatted):
                judgment_text = judgment_text[len(prompt_formatted):]
            elif prompt and judgment_text.startswith(prompt):
                 judgment_text = judgment_text[len(prompt):]
            
            judgment_text = judgment_text.strip()
            lines = [line.strip() for line in judgment_text.splitlines() if line.strip()]

            score = None
            if lines:
                score = self._extract_numeric_score(lines[0])

            explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else judgment_text
            correct = bool(score is not None and score >= 4.0)
            confidence = self._confidence_from_score(score)

            logger.info("LLM Judge - Score: %s, Correct: %s", score, correct)

            return {
                'correct': correct,
                'confidence': confidence,
                'explanation': explanation or judgment_text,
                'score': score
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

    def _token_overlap_ratio(self, text_a: str, text_b: str) -> float:
        """Compute symmetric token-overlap ratio between two texts."""
        if not text_a or not text_b:
            return 0.0
        tokens_a = set(self._tokenize(text_a))
        tokens_b = set(self._tokenize(text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        denom = max(len(tokens_a), len(tokens_b))
        return len(intersection) / denom if denom else 0.0

    def _extract_numeric_score(self, text: str) -> Optional[float]:
        """
        Extract the numeric score (1-5) from the text.
        Prioritizes numbers at the beginning of the text or explicitly labeled.
        """
        if not text:
            return None
            
        # 1. Try to match number at start of line (most common/correct format per instructions)
        # matches "5", "5.0", "4.5"
        match = re.search(r"^\s*(\d+(?:\.\d+)?)\b", text)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 5.0:  # Allow 0-5
                    return val
            except ValueError:
                pass

        # 2. Try to match "Score: X" or "Rating: X" anywhere
        match = re.search(r"(?:score|rating|grade)\s*[:=]\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 5.0:
                    return val
            except ValueError:
                pass
                
        # 3. Try "score of X" or "give it a X"
        match = re.search(r"\b(?:give|assigned|received)\s+(?:it\s+)?(?:a\s+)?score\s+(?:of\s+)?(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 5.0:
                    return val
            except ValueError:
                pass
        
        # NOTE: We intentionally do NOT fall back to finding *any* number in the string,
        # as that often picks up years (2020), percentages, or monetary values mentioned 
        # in the explanation if the model fails to output a score at the start.
        
        return None

    def _confidence_from_score(self, score: Optional[float]) -> Optional[str]:
        """Map numeric score to coarse confidence bucket."""
        if score is None:
            return None
        if score >= 4.5:
            return "HIGH"
        if score >= 3.5:
            return "MEDIUM"
        return "LOW"
    
    def evaluate_generation(
        self,
        prediction: str,
        reference: str,
        question: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        gold_contexts: Optional[List[str]] = None,
        langchain_llm: Any = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated text
        
        Args:
            prediction: Generated text
            reference: Reference text
            question: Original question (optional, for LLM judge)
            contexts: Retrieved context strings supplied to the generator
            gold_contexts: Gold/reference context strings for diagnostics
            langchain_llm: LangChain-compatible LLM wrapper (for RAGAS)
            
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

        # RAGAS-style holistic metrics
        if self.use_ragas and self._ragas_metrics:
            ragas_scores = self.compute_ragas_scores(
                question=question or "",
                prediction=prediction,
                contexts=contexts or [],
                reference_answer=reference,
                reference_contexts=gold_contexts or [],
                langchain_llm=langchain_llm,
            )
            if ragas_scores:
                results['ragas'] = ragas_scores
        
        # Log summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"  BLEU-4: {results['bleu'].get('bleu_4', 0):.4f}")
        logger.info(f"  ROUGE-L F1: {results['rouge'].get('rouge_l_f1', 0):.4f}")
        if 'bertscore' in results and results['bertscore']['f1']:
            logger.info(f"  BERTScore F1: {results['bertscore']['f1']:.4f}")
        if 'ragas' in results:
            for metric_name, score in results['ragas'].items():
                logger.info(f"  RAGAS {metric_name}: {score:.4f}")
        
        logger.info("=" * 80)
        
        return results
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        questions: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        gold_contexts: Optional[List[List[str]]] = None,
        langchain_llm: Any = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            questions: List of questions (optional)
            contexts: List of retrieved context lists per sample
            gold_contexts: List of gold context lists per sample
            langchain_llm: LangChain-compatible LLM wrapper (for RAGAS)
            
        Returns:
            Dictionary with aggregated metrics
        """
        logger.info(f"Evaluating batch of {len(predictions)} predictions")
        
        all_results = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            question = questions[i] if questions else None
            sample_contexts = contexts[i] if contexts and i < len(contexts) else None
            sample_gold_contexts = (
                gold_contexts[i] if gold_contexts and i < len(gold_contexts) else None
            )
            result = self.evaluate_generation(
                pred,
                ref,
                question,
                contexts=sample_contexts,
                gold_contexts=sample_gold_contexts,
                langchain_llm=langchain_llm,
            )
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
            scores = [r['bleu'][key] for r in results if 'bleu' in r and r['bleu'] and key in r['bleu']]
            if scores:
                aggregated[f'{key}_mean'] = np.mean(scores)
                aggregated[f'{key}_std'] = np.std(scores)
        
        # Aggregate ROUGE scores
        rouge_metrics = ['rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1']
        for metric in rouge_metrics:
            scores = [r['rouge'][metric] for r in results if 'rouge' in r and r['rouge'] and metric in r['rouge']]
            if scores:
                aggregated[f'{metric}_mean'] = np.mean(scores)
                aggregated[f'{metric}_std'] = np.std(scores)
        
        # Aggregate BERTScore
        if any('bertscore' in r for r in results):
            for metric in ['precision', 'recall', 'f1']:
                scores = [r['bertscore'][metric] for r in results 
                         if 'bertscore' in r and r['bertscore'] and r['bertscore'].get(metric) is not None]
                if scores:
                    aggregated[f'bertscore_{metric}_mean'] = np.mean(scores)
                    aggregated[f'bertscore_{metric}_std'] = np.std(scores)

        # Aggregate RAGAS scores
        ragas_metric_names: set = set()
        for r in results:
            if 'ragas' in r and r['ragas']:
                ragas_metric_names.update(r['ragas'].keys())
        for metric_name in ragas_metric_names:
            scores = [
                r['ragas'][metric_name]
                for r in results
                if 'ragas' in r and r['ragas'] and metric_name in r['ragas']
            ]
            if scores:
                aggregated[f'ragas_{metric_name}_mean'] = float(np.mean(scores))
                aggregated[f'ragas_{metric_name}_std'] = float(np.std(scores))

        # Aggregate LLM judge scores
        judge_scores: List[float] = []
        judge_correct_flags: List[bool] = []
        for r in results:
            judge = r.get('llm_judge') if isinstance(r, dict) else None
            if not judge:
                continue
            score = judge.get('score')
            if isinstance(score, (int, float)):
                judge_scores.append(float(score))
            correct = judge.get('correct')
            if isinstance(correct, bool):
                judge_correct_flags.append(correct)
        if judge_scores:
            aggregated['llm_judge_score_mean'] = float(np.mean(judge_scores))
            aggregated['llm_judge_score_std'] = float(np.std(judge_scores))
        if judge_correct_flags:
            aggregated['llm_judge_accuracy'] = float(np.mean(judge_correct_flags))
        
        return aggregated

    def _aggregate_retrieval_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate retrieval metrics"""
        aggregated = {}
        if not results:
            return aggregated
            
        keys = [
            "num_retrieved", "exact_match", "max_token_overlap", "mean_token_overlap",
            "doc_hit_at_k", "page_hit_at_k", "chunk_hit_at_k",
            "doc_recall_at_k", "page_recall_at_k", "chunk_recall_at_k",
            "doc_hit_rank", "page_hit_rank", "chunk_hit_rank"
        ]
        
        for key in keys:
            # Filter None values (e.g. rank can be None)
            vals = [r[key] for r in results if r.get(key) is not None]
            if vals:
                aggregated[f"retrieval_{key}_mean"] = float(np.mean(vals))
                # Skip std for binary hit flags if desired, but including for completeness is fine
                aggregated[f"retrieval_{key}_std"] = float(np.std(vals))
        
        # Add failure reason counts
        reasons = [r.get("failure_reason") for r in results if r.get("failure_reason")]
        if reasons:
            total = len(reasons)
            counts = Counter(reasons)
            for reason, count in counts.items():
                aggregated[f"retrieval_fail_{reason}_pct"] = count / total
                
        return aggregated

    def _aggregate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Helper to aggregate both generation and retrieval metrics for a list of samples"""
        # Extract generation metrics list (ensure not None)
        gen_metrics = [s.get("generation_evaluation", {}) for s in samples if s.get("generation_evaluation")]
        # Extract retrieval metrics list
        ret_metrics = [s.get("retrieval_evaluation", {}) for s in samples if s.get("retrieval_evaluation")]
        
        agg_gen = self._aggregate_results(gen_metrics)
        agg_ret = self._aggregate_retrieval_results(ret_metrics)
        
        return {**agg_gen, **agg_ret}

    def summarize_experiment(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across all samples, and by question_type/reasoning.
        """
        # 1. Full aggregation
        full_agg = self._aggregate_samples(samples)
        
        # 2. By question_type
        by_type = {}
        # Get all unique types, filtering out None
        types = set(str(s.get("question_type")) for s in samples if s.get("question_type"))
        for t in types:
            subset = [s for s in samples if str(s.get("question_type")) == t]
            by_type[t] = self._aggregate_samples(subset)
            
        # 3. By question_reasoning
        by_reasoning = {}
        reasons = set(str(s.get("question_reasoning")) for s in samples if s.get("question_reasoning"))
        for r in reasons:
            subset = [s for s in samples if str(s.get("question_reasoning")) == r]
            by_reasoning[r] = self._aggregate_samples(subset)
            
        return {
            "overall": full_agg,
            "by_question_type": by_type,
            "by_question_reasoning": by_reasoning
        }



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