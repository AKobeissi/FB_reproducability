"""
Re-generate answers with a more concise prompt for existing predictions.
Takes retrieved chunks and generates new answers with a direct, concise prompt.
"""
import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Any
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Concise prompt template
CONCISE_PROMPT_TEMPLATE = """You are a financial analyst. Answer the question directly and concisely using only the information in the context below.

Context:
{context}

Question: {question}

Instructions:
- Provide a direct, factual answer
- Do not show your reasoning or thought process
- Do not add explanations unless specifically asked
- Be precise and to the point
- If the answer is a number, provide it clearly

Answer:"""

def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
    """Load the language model."""
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    return tokenizer, model

def generate_answer(question: str, context: str, tokenizer, model, max_new_tokens: int = 256):
    """Generate answer with concise prompt."""
    prompt = CONCISE_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    # Prepare for chat models
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    if not text:
        return []
    clean_text = re.sub(r'[,$]', '', str(text))
    matches = re.findall(r'-?\d*\.?\d+', clean_text)
    try:
        return [float(m) for m in matches]
    except ValueError:
        return []

def check_numeric_match(reference: str, prediction: str) -> float:
    """Check if any number in prediction matches any in reference."""
    ref_nums = extract_numbers(reference)
    pred_nums = extract_numbers(prediction)
    
    if not ref_nums or not pred_nums:
        return 0.0
    
    for r_num in ref_nums:
        for p_num in pred_nums:
            if np.isclose(r_num, p_num, atol=0.05, rtol=0.05):
                return 1.0
    return 0.0

def compute_bleu(reference: str, prediction: str) -> float:
    """Compute BLEU score."""
    if not reference or not prediction:
        return 0.0
    
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    
    if not ref_tokens or not pred_tokens:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    try:
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except Exception:
        return 0.0

def compute_rouge(reference: str, prediction: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
    if not reference or not prediction:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def regenerate_predictions(input_file: str, output_file: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Re-generate answers for all predictions with concise prompt."""
    
    # Load predictions
    logger.info(f"Loading predictions from {input_file}")
    with open(input_file, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(model_name, device)
    
    # Process each prediction
    results = []
    
    for i, pred in enumerate(tqdm(predictions, desc="Regenerating answers")):
        # Extract context from retrieved chunks
        chunks = pred.get('retrieved_chunks', [])
        context_text = "\n\n".join([chunk['text'] for chunk in chunks])
        
        question = pred.get('question', '')
        reference_answer = pred.get('answer', '')
        old_model_answer = pred.get('model_answer', '')
        
        # Generate new answer
        try:
            generated_answer = generate_answer(question, context_text, tokenizer, model)
        except Exception as e:
            logger.warning(f"Generation failed for sample {i}: {e}")
            generated_answer = ""
        
        # Compute metrics
        numeric_match = check_numeric_match(reference_answer, generated_answer)
        bleu_score = compute_bleu(reference_answer, generated_answer)
        rouge_scores = compute_rouge(reference_answer, generated_answer)
        
        # Build result
        result = {
            'financebench_id': pred.get('financebench_id'),
            'company': pred.get('company'),
            'doc_name': pred.get('doc_name'),
            'question': question,
            'reference_answer': reference_answer,
            'old_model_answer': old_model_answer,
            'new_generated_answer': generated_answer,
            'retrieved_chunks': chunks,
            'metrics': {
                'numeric_match': numeric_match,
                'bleu': bleu_score,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL']
            }
        }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(predictions)} samples")
    
    # Compute aggregate metrics
    all_numeric = [r['metrics']['numeric_match'] for r in results]
    all_bleu = [r['metrics']['bleu'] for r in results]
    all_rouge1 = [r['metrics']['rouge1'] for r in results]
    all_rouge2 = [r['metrics']['rouge2'] for r in results]
    all_rougeL = [r['metrics']['rougeL'] for r in results]
    
    aggregate_metrics = {
        'numeric_match': np.mean(all_numeric),
        'bleu': np.mean(all_bleu),
        'rouge1': np.mean(all_rouge1),
        'rouge2': np.mean(all_rouge2),
        'rougeL': np.mean(all_rougeL),
        'total_samples': len(results)
    }
    
    # Save results
    output_data = {
        'aggregate_metrics': aggregate_metrics,
        'predictions': results,
        'model_info': {
            'model_name': model_name,
            'prompt_template': CONCISE_PROMPT_TEMPLATE
        }
    }
    
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print aggregate metrics
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE METRICS:")
    logger.info(f"  Numeric Match: {aggregate_metrics['numeric_match']:.4f}")
    logger.info(f"  BLEU:          {aggregate_metrics['bleu']:.4f}")
    logger.info(f"  ROUGE-1:       {aggregate_metrics['rouge1']:.4f}")
    logger.info(f"  ROUGE-2:       {aggregate_metrics['rouge2']:.4f}")
    logger.info(f"  ROUGE-L:       {aggregate_metrics['rougeL']:.4f}")
    logger.info("="*80)
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-generate answers with concise prompt")
    parser.add_argument('input_file', type=str, help='Input predictions JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: input_file with _regenerated suffix)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model to use for generation')
    
    args = parser.parse_args()
    
    # Default output file
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_regenerated.json")
    
    regenerate_predictions(args.input_file, args.output, args.model)
