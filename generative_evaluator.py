
import logging
import json
import argparse
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import Counter
from types import SimpleNamespace
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mandatory Dependencies - Fail if missing
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
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

# Optional Imports for LLM Initialization
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN_OPENAI = True
except ImportError:
    HAS_LANGCHAIN_OPENAI = False

try:
    from langchain_huggingface import HuggingFacePipeline as LangchainHFPipeline
    HAS_LANGCHAIN_HF = True
except ImportError:
    HAS_LANGCHAIN_HF = False


# Use provided LlamaIndex templates directly as fallback/default
CORRECTNESS_SYS_TMPL = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query,
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.
"""

CORRECTNESS_USER_TMPL = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

class OpenAIChatPipeline:
    """Adapter that mimics the transformers pipeline interface for OpenAI chat models."""

    def __init__(self, client: Any, model_id: str):
        self._client = client
        self._model_id = model_id
        self.tokenizer = SimpleNamespace(eos_token_id=None)

    def __call__(self, prompt: str, max_new_tokens: int = 200, **_: Any) -> List[Dict[str, str]]:
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": "You are an impartial grader for FinanceBench answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        text = response.choices[0].message.content or ""
        generated_text = f"{prompt}\n{text.strip()}"
        return [{"generated_text": generated_text}]


class GenerativeEvaluator:
    """
    Evaluator for Generative aspects of RAG systems.
    Computes BLEU, ROUGE, BERTScore, LLM Judge, and RAGAS metrics.
    """

    def __init__(
        self,
        use_bertscore: bool = True,
        use_llm_judge: bool = True,
        use_ragas: bool = True,
        ragas_embedding_model: Optional[str] = None,
        ragas_device: Optional[str] = None,
        judge_pipeline: Any = None
    ):
        self.use_bertscore = use_bertscore
        self.use_llm_judge = use_llm_judge
        self.use_ragas = use_ragas
        
        # BERTScore
        self.bertscore_scorer = None
        if self.use_bertscore:
            try:
                self.bertscore_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                logger.info("BERTScore initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize BERTScore: {e}")
                self.use_bertscore = False

        # ROUGE
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # BLEU
        self.sacrebleu = sacrebleu

        # LLM Judge
        self.judge_pipeline = judge_pipeline
        self._judge_max_new_tokens = 200
        
        # RAGAS
        self._ragas_embedding_model_name = ragas_embedding_model
        self._ragas_device = ragas_device
        self._ragas_embeddings = None
        self._ragas_llm_cache = {}
        self._ragas_metrics = []
        if self.use_ragas:
            self._ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            
    # --- BLEU ---
    def compute_bleu(self, prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
        if self.sacrebleu:
            try:
                # Use sacrebleu for reliable scoring
                score = self.sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0
                return {'bleu_4': score} 
            except Exception:
                pass
        return self._compute_bleu_fallback(prediction, reference, max_n)

    def _compute_bleu_fallback(self, prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        scores = {}
        for n in range(1, max_n + 1):
            if len(pred_tokens) < n:
                scores[f'bleu_{n}'] = 0.0
                continue
            pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
            
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            matches = sum((pred_counter & ref_counter).values())
            precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            
            # Brevity penalty
            bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
            scores[f'bleu_{n}'] = bp * precision
        return scores

    # --- ROUGE ---
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        if self.rouge_scorer:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge_1_f1': scores['rouge1'].fmeasure,
                'rouge_2_f1': scores['rouge2'].fmeasure,
                'rouge_l_f1': scores['rougeL'].fmeasure
            }
        return self._compute_rouge_fallback(prediction, reference)

    def _compute_rouge_fallback(self, prediction: str, reference: str) -> Dict[str, float]:
        # Simple fallback for ROUGE-L (LCS based)
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        m, n = len(pred_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return {'rouge_l_f1': 0.0}
            
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        prec = lcs / m
        rec = lcs / n
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {'rouge_l_f1': f1}

    # --- BERTScore ---
    def compute_bertscore(self, prediction: str, reference: str) -> Dict[str, float]:
        if not self.use_bertscore or not self.bertscore_scorer:
            return {}
        try:
            P, R, F1 = self.bertscore_scorer.score([prediction], [reference])
            return {
                'bertscore_precision': P.item(),
                'bertscore_recall': R.item(),
                'bertscore_f1': F1.item()
            }
        except Exception as e:
            logger.error(f"BERTScore error: {e}")
            return {}

    # --- LLM Judge ---
    def llm_judge(self, question: str, prediction: str, reference: str) -> Dict[str, Any]:
        if not self.use_llm_judge or not self.judge_pipeline:
            return {}
        
        prompt = self._create_judge_prompt(question, prediction, reference)
        try:
            # Handling pipeline call similar to original
            # This is a simplification; assumes judge_pipeline is a callable HF pipeline or similar adapter
            if hasattr(self.judge_pipeline, "tokenizer") and hasattr(self.judge_pipeline.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                prompt_formatted = self.judge_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_formatted = f"<s>[INST] {prompt} [/INST]"

            response = self.judge_pipeline(
                prompt_formatted,
                max_new_tokens=self._judge_max_new_tokens,
                return_full_text=False
            )
            
            # Extract text
            if isinstance(response, list) and len(response) > 0:
                 text = response[0].get('generated_text', '')
            else:
                 text = str(response)

            score = self._extract_score(text)
            return {
                'llm_judge_score': score,
                'llm_judge_explanation': text,
                'llm_judge_correct': score >= 4.0 if score else None
            }
        except Exception as e:
            logger.error(f"LLM Judge error: {e}")
            return {}

    def _create_judge_prompt(self, question: str, prediction: str, reference: str) -> str:
        system = CORRECTNESS_SYS_TMPL
        user_tpl = CORRECTNESS_USER_TMPL
        user_msg = user_tpl.format(query=question, reference_answer=reference, generated_answer=prediction)
        return f"{system}\n\n{user_msg}\n\nReturn the numeric score (1-5) on the first line and reasoning on the next."

    def _extract_score(self, text: str) -> Optional[float]:
        match = re.search(r"^\s*(\d+(?:\.\d+)?)\b", text)
        if match:
            return float(match.group(1))
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    # --- RAGAS ---
    def compute_ragas(self, question: str, prediction: str, contexts: List[str], 
                      reference: str, gold_contexts: List[str], langchain_llm: Any) -> Dict[str, float]:
        if not self.use_ragas:
            return {}
        
        # Initialize embeddings if needed
        if not self._ragas_embeddings:
            model_name = self._ragas_embedding_model_name or "sentence-transformers/all-mpnet-base-v2"
            try:
                self._ragas_embeddings = RagasHuggingFaceEmbeddings(
                    model=model_name,
                    device=self._ragas_device
                )
            except Exception as e:
                logger.error(f"RAGAS embeddings init failed: {e}")
                return {}

        # Wrap LLM
        ragas_llm = None
        if langchain_llm:
            cache_key = id(langchain_llm)
            if cache_key in self._ragas_llm_cache:
                ragas_llm = self._ragas_llm_cache[cache_key]
            else:
                try:
                    ragas_llm = LangchainLLMWrapper(langchain_llm)
                    self._ragas_llm_cache[cache_key] = ragas_llm
                except Exception as e:
                    logger.warning(f"RAGAS LLM wrap failed: {e}")
        
        if not ragas_llm:
            return {}

        sample = {
            "user_input": question or "",
            "response": prediction or "",
            "retrieved_contexts": contexts or [],
            "reference": reference or "",
            "reference_contexts": gold_contexts or []
        }
        
        try:
            dataset = EvaluationDataset.from_list([sample])
            results = ragas_evaluate(
                dataset,
                metrics=self._ragas_metrics,
                llm=ragas_llm,
                embeddings=self._ragas_embeddings
            )
            return {f"ragas_{k}": v for k, v in results.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}")
            return {}

    # --- Utils ---
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def evaluate_sample(self, sample: Dict[str, Any], langchain_llm: Any = None) -> Dict[str, Any]:
        pred = sample.get("generated_answer", "")
        ref = sample.get("reference_answer", "")
        q = sample.get("question", "")
        contexts = [c.get("text", "") for c in sample.get("retrieved_chunks", []) if c.get("text")]
        
        # Gold contexts
        gold_contexts = []
        golds = sample.get("gold_evidence_segments", [])
        if isinstance(golds, list):
            for g in golds:
                if isinstance(g, dict) and g.get("text"):
                    gold_contexts.append(g["text"])
        
        metrics = {}
        metrics.update(self.compute_bleu(pred, ref))
        metrics.update(self.compute_rouge(pred, ref))
        metrics.update(self.compute_bertscore(pred, ref))
        metrics.update(self.llm_judge(q, pred, ref))
        metrics.update(self.compute_ragas(q, pred, contexts, ref, gold_contexts, langchain_llm))
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Run Generative Evaluation on RAG outputs.")
    parser.add_argument("input_file", help="Path to input JSON file.")
    parser.add_argument("--output-dir", help="Directory to save evaluation results.")
    parser.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM Judge")
    parser.add_argument("--no-ragas", action="store_true", help="Disable RAGAS")
    
    # LLM Judge Configuration
    parser.add_argument("--judge-model", default="openai/gpt-4o", help="Model to use for LLM judge")
    parser.add_argument("--judge-provider", choices=["huggingface", "openai"], default="auto", 
                        help="Provider for LLM judge (auto: detect from model name)")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY", help="Env var for OpenAI API key")
    parser.add_argument("--openai-api-base", default="https://api.openai.com/v1", help="Base URL for OpenAI")
    parser.add_argument("--device", default="auto", help="Device for local models (cpu, cuda, auto)")
    
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and "results" in data:
        samples = data["results"]
    else:
        logger.error("Invalid JSON format.")
        return

    # Initialize Judge Pipeline if requested
    judge_pipeline = None
    langchain_llm = None
    
    use_llm_judge = not args.no_llm_judge
    use_ragas = not args.no_ragas
    
    if use_llm_judge or use_ragas:
        provider = args.judge_provider
        if provider == "auto":
            if "gpt" in args.judge_model.lower() or "openai" in args.judge_model.lower():
                provider = "openai"
            else:
                provider = "huggingface"
        
        logger.info(f"Initializing LLM Judge/RAGAS using {provider} provider and model {args.judge_model}")
        
        if provider == "openai":
            if not HAS_OPENAI:
                logger.error("openai package required for OpenAI judge. Install with: pip install openai")
                use_llm_judge = False
                use_ragas = False
            else:
                api_key = os.environ.get(args.openai_api_key_env)
                if not api_key:
                    logger.error(f"API key not found in {args.openai_api_key_env}")
                    use_llm_judge = False
                    use_ragas = False
                else:
                    try:
                        client = OpenAI(api_key=api_key, base_url=args.openai_api_base)
                        judge_pipeline = OpenAIChatPipeline(client, args.judge_model)
                        
                        if use_ragas and HAS_LANGCHAIN_OPENAI:
                            langchain_llm = ChatOpenAI(
                                model=args.judge_model,
                                api_key=api_key,
                                base_url=args.openai_api_base,
                                temperature=0.0
                            )
                    except Exception as e:
                        logger.error(f"Failed to initialize OpenAI client: {e}")
                        use_llm_judge = False
                        use_ragas = False

        elif provider == "huggingface":
            if not HAS_TRANSFORMERS:
                logger.error("transformers package required for HuggingFace judge.")
                use_llm_judge = False
                use_ragas = False
            else:
                try:
                    logger.info(f"Loading local model: {args.judge_model}")
                    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
                    model_kwargs = {"device_map": args.device} if args.device != "cpu" else {}
                    if args.device == "cuda":
                        model_kwargs["torch_dtype"] = torch.float16
                        
                    model = AutoModelForCausalLM.from_pretrained(args.judge_model, **model_kwargs)
                    judge_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=200,
                    )
                    
                    if use_ragas and HAS_LANGCHAIN_HF:
                        langchain_llm = LangchainHFPipeline(pipeline=judge_pipeline)
                except Exception as e:
                    logger.error(f"Failed to initialize local model: {e}")
                    use_llm_judge = False
                    use_ragas = False

    evaluator = GenerativeEvaluator(
        use_bertscore=not args.no_bertscore,
        use_llm_judge=use_llm_judge,
        use_ragas=use_ragas,
        judge_pipeline=judge_pipeline
    )

    results = []
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            logger.info(f"Processing sample {i}/{len(samples)}")
        metrics = evaluator.evaluate_sample(sample, langchain_llm=langchain_llm)
        sample["generative_metrics"] = metrics
        results.append(sample)

    # Calculate averages
    agg_metrics = {}
    metric_keys = set()
    for r in results:
        metric_keys.update(r.get("generative_metrics", {}).keys())
    
    for k in metric_keys:
        vals = [r["generative_metrics"][k] for r in results if r["generative_metrics"].get(k) is not None]
        if vals:
             # handle booleans (llm_judge_correct)
            if all(isinstance(v, bool) for v in vals):
                agg_metrics[f"avg_{k}"] = sum(vals) / len(vals)
            elif all(isinstance(v, (int, float)) for v in vals):
                agg_metrics[f"avg_{k}"] = sum(vals) / len(vals)

    print("\nGenerative Evaluation Results (Averages):")
    print(json.dumps(agg_metrics, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{input_path.stem}_generative_eval.json"
        with open(out_file, 'w') as f:
            # Save full results including per-sample metrics
            json.dump({"summary": agg_metrics, "results": results}, f, indent=2)
        logger.info(f"Saved evaluation results to {out_file}")

if __name__ == "__main__":
    main()
