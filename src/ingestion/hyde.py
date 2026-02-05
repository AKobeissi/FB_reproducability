"""
HyDE (Hypothetical Document Embeddings) Module

Generates hypothetical documents for query reformulation.
Used at inference time to improve retrieval quality.
"""

import logging
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


def generate_hypothetical_documents(
    query: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_generations: int = 1,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    device: Optional[str] = None
) -> List[str]:
    """
    Generate hypothetical documents that answer the given query.
    
    Args:
        query: The input question/query
        model_name: HuggingFace model to use for generation
        num_generations: Number of hypothetical documents to generate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more diverse)
        device: Device to run on (None = auto-detect)
        
    Returns:
        List of generated hypothetical documents
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Generating {num_generations} hypothetical document(s) with {model_name}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        # Finance-specific HyDE prompt
        hyde_prompt = (
            f"Please write a short financial report excerpt (1 paragraph) that precisely answers the question below. "
            f"Do not introduce the text, just write the excerpt.\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )
        
        hypotheticals = []
        
        for i in range(num_generations):
            # Format as chat message if using instruct model
            if "Instruct" in model_name or "instruct" in model_name:
                messages = [{"role": "user", "content": hyde_prompt}]
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                text = hyde_prompt
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if num_generations > 1 else 0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and extract generated text (remove prompt)
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to extract just the generated portion
            if "Passage:" in full_text:
                generated_text = full_text.split("Passage:")[-1].strip()
            else:
                # Fallback: remove input prompt
                generated_text = full_text[len(text):].strip()
            
            hypotheticals.append(generated_text)
            logger.debug(f"Generated hypothesis {i+1}/{num_generations}: {generated_text[:100]}...")
        
        # Clean up
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return hypotheticals
        
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        # Fallback to original query
        return [query] * num_generations
