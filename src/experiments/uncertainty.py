"""
Uncertainty Quantification Experiment (MC-Dropout + MI).
Implements the pipeline for:
1. Retrieval (Candidates)
2. Deterministic Reranking
3. MC-Dropout on Top-L Candidates
4. Mutual Information (MI) & Confidence Estimation
5. Generation with Selective Answering Signals
"""
import logging
import torch
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# Internal imports
from src.retrieval.vectorstore import (
    build_chroma_store, 
    get_chroma_db_path, 
    populate_chroma_store, 
    save_store_config
)
from src.ingestion.pdf_utils import load_pdf_with_fallback
from src.experiments.rag_shared_vector import _get_doc_text, _log_pdf_sources

logger = logging.getLogger(__name__)

# --- CONFIGURATION DEFAULTS ---
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_KCAND = 100
DEFAULT_L_MC = 30    # Number of chunks to run MC dropout on
DEFAULT_T = 10       # Number of MC passes
DEFAULT_ALPHA = 1.0  # Scaling factor for sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- PART C2 & C3: MC-DROPOUT & MI COMPUTATION ---

def mc_scores(reranker: CrossEncoder, query: str, chunks: List[str], T: int) -> List[List[float]]:
    """
    Run MC-Dropout on the reranker model for the given query and chunks.
    """
    if not chunks:
        return []

    # Prepare inputs
    pairs = [[query, c] for c in chunks]
    
    device = reranker.model.device
    tokenizer = reranker.tokenizer
    model = reranker.model
    
    # Tokenize once
    features = tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=reranker.max_length
    ).to(device)

    chunk_scores_t = [] # Will store T arrays of size [num_chunks]

    # Run T passes with Dropout ACTIVE
    with torch.no_grad():
        for i in range(T):
            model.train() # <--- CRITICAL: Enable Dropout
            
            outputs = model(**features)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            
            # Squeeze to get shape [batch_size]
            scores = logits.squeeze(-1).cpu().numpy()
            chunk_scores_t.append(scores)

    # Transpose result to [num_chunks, T]
    result = np.array(chunk_scores_t).T.tolist()
    return result

def compute_mi(scores_t: List[float], alpha: float = 1.0) -> Dict[str, float]:
    """
    Compute Mu, Sigma, MI, and P_bar for a single chunk's distribution of scores.
    """
    scores = np.array(scores_t)
    
    # 1. Basic Stats
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    
    # 2. Map to Probabilities (Bernoulli proxy)
    probs = sigmoid(alpha * scores)
    
    # Clamp to avoid log(0)
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    
    # 3. MI Computation
    def binary_entropy(p):
        return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    
    p_bar = np.mean(probs)
    entropy_of_mean = binary_entropy(p_bar)
    mean_of_entropies = np.mean(binary_entropy(probs))
    
    mi = entropy_of_mean - mean_of_entropies
    
    return {
        "mu": mu,
        "sigma": sigma,
        "mi": float(mi),
        "p_bar": float(p_bar),
        "entropy_of_mean": float(entropy_of_mean),
        "mean_of_entropies": float(mean_of_entropies)
    }

# --- EXPERIMENT PIPELINE ---

def run_uncertainty_experiment(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main pipeline for S4 (Uncertainty) Experiment.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING UNCERTAINTY EXPERIMENT (MC-Dropout)")
    logger.info(f"T={experiment.mc_t} | L_mc={experiment.mc_l} | Alpha={experiment.mc_alpha}")
    logger.info("=" * 80)

    # --- 1. Setup Retrieval & Ingestion ---
    
    # A. Validate PDF Directory
    if not experiment.pdf_local_dir or not os.path.exists(experiment.pdf_local_dir):
         logger.warning(f"PDF dir {experiment.pdf_local_dir} invalid. Trying auto-detection...")
         experiment.pdf_local_dir = Path(os.getcwd()) / "pdfs"
         if not os.path.exists(experiment.pdf_local_dir):
             logger.error("Could not find PDF directory. Ingestion will fail.")
    
    # B. Initialize Vector Store (Lazy)
    # This creates the object. If the directory is empty, we must populate it.
    base_retriever, vectordb, is_new = build_chroma_store(experiment, "all", lazy_load=True)
    
    # FIX: Check strictly for None. An empty Chroma DB might evaluate to False.
    if vectordb is None:
        logger.error("Failed to initialize vector store object (returned None).")
        return []

    # C. Ingestion Loop (Populate if needed)
    # Identify unique docs needed
    unique_docs = {}
    for sample in data:
        unique_docs.setdefault(sample.get('doc_name', 'unknown'), sample.get('doc_link', ''))
        
    # Check what is already in the store
    _, db_path = get_chroma_db_path(experiment, "all")
    meta_path = os.path.join(db_path, "shared_meta.json")
    available_docs = set()
    pdf_source_map = {}

    if not is_new and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                available_docs = set(meta.get("available_docs", []))
                pdf_source_map = meta.get("pdf_source_map", {})
        except Exception: 
            pass

    # Calculate missing docs
    docs_to_process = {k: v for k, v in unique_docs.items() if k not in available_docs}
    
    if docs_to_process:
        logger.info(f"Ingesting {len(docs_to_process)} missing documents into Vector Store...")
        
        for doc_name, doc_link in tqdm(docs_to_process.items(), desc="Ingesting PDFs"):
            pdf_docs, src = load_pdf_with_fallback(
                doc_name, 
                doc_link, 
                getattr(experiment, 'pdf_local_dir', None)
            )
            pdf_source_map[doc_name] = src
            
            if pdf_docs:
                # Chunk the PDF
                chunks = experiment._chunk_text_langchain(pdf_docs, metadata={'doc_name': doc_name})
                if chunks:
                    # Insert into Chroma
                    populate_chroma_store(experiment, vectordb, chunks, "all")
                    available_docs.add(doc_name)
        
        # Update metadata on disk
        save_store_config(experiment, db_path)
        with open(meta_path, 'w') as f:
            json.dump({"available_docs": list(available_docs), "pdf_source_map": pdf_source_map}, f)
            
    # --- 2. Load Reranker ---
    device = experiment.device
    reranker_model_id = getattr(experiment, "reranker_model", DEFAULT_RERANKER)
    logger.info(f"Loading Reranker: {reranker_model_id}")
    reranker = CrossEncoder(
        reranker_model_id,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
        device=device,
        trust_remote_code=True
    )

    results = []
    
    # Create debug directory for sample outputs
    debug_dir = Path(experiment.output_dir) / "debug_samples"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Main Loop ---
    for i, sample in enumerate(tqdm(data, desc="Uncertainty Pipeline")):
        query = sample.get('question', '')
        if not query: continue
        
        doc_name = sample.get('doc_name', 'unknown')
        
        # A. Retrieve Candidates
        K_cand = getattr(experiment, "k_cand", DEFAULT_KCAND)
        try:
            initial_docs = vectordb.similarity_search(query, k=K_cand)
        except Exception as e:
            logger.warning(f"Retrieval failed for sample {i}: {e}")
            initial_docs = []
        
        if not initial_docs:
            logger.warning(f"No documents found for sample {i}")
            results.append({
                'sample_id': i,
                'doc_name': doc_name,
                'doc_link': sample.get('doc_link', ''),
                'question': query,
                'question_type': sample.get('question_type', ''),
                'generated_answer': "I cannot answer this question as no relevant documents were retrieved.",
                'generation_length': len("I cannot answer this question as no relevant documents were retrieved."),
                'context_length': 0,
                'confidence_margin': 0.0,
                'confidence_mi': 0.0,
                'confidence_sigma': 0.0,
                'retrieved_chunks': [],
                'gold_evidence': '',
                'gold_evidence_segments': []
            })
            continue

        # B. Deterministic Reranking
        doc_texts = [_get_doc_text(d) for d in initial_docs]
        pairs = [[query, txt] for txt in doc_texts]
        
        det_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        
        for d, s in zip(initial_docs, det_scores):
            d.metadata['det_score'] = float(s)
            
        initial_docs.sort(key=lambda x: x.metadata['det_score'], reverse=True)
        
        # C. Select Top L_mc for Uncertainty
        L_mc = experiment.mc_l
        top_L_docs = initial_docs[:L_mc]
        top_L_texts = [_get_doc_text(d) for d in top_L_docs]
        
        # D. MC-Dropout Loop
        mc_matrix = mc_scores(reranker, query, top_L_texts, T=experiment.mc_t)
        
        # E. Compute Statistics
        enhanced_docs = []
        for rank_idx, (doc, scores_t) in enumerate(zip(top_L_docs, mc_matrix)):
            stats = compute_mi(scores_t, alpha=experiment.mc_alpha)
            doc.metadata.update(stats)
            enhanced_docs.append(doc)
            
        # F. Evidence Selection (Top N)
        top_k = experiment.top_k
        final_evidence_docs = enhanced_docs[:top_k]
        
        # G. Compute Confidence Signals
        margin_conf = 0.0
        if len(enhanced_docs) >= 2:
            s1 = enhanced_docs[0].metadata['det_score']
            s2 = enhanced_docs[1].metadata['det_score']
            margin_conf = float(s1 - s2)
        elif len(enhanced_docs) == 1:
            margin_conf = float(enhanced_docs[0].metadata['det_score'])

        mi_values = [d.metadata['mi'] for d in final_evidence_docs]
        conf_mi = -float(np.mean(mi_values)) if mi_values else 0.0
        
        sigma_values = [d.metadata['sigma'] for d in final_evidence_docs]
        conf_sigma = -float(np.mean(sigma_values)) if sigma_values else 0.0

        # H. Generation
        context_text = "\n\n".join([_get_doc_text(d) for d in final_evidence_docs])
        ans, final_prompt = experiment._generate_answer(query, context_text, return_prompt=True)
        
        # I. Save Results
        gold_segs, gold_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
        
        retrieved_chunks = []
        for r, d in enumerate(final_evidence_docs):
            retrieved_chunks.append({
                "rank": r + 1,
                "text": _get_doc_text(d),
                "det_score": d.metadata.get('det_score'),
                "mi": d.metadata.get('mi'),
                "sigma": d.metadata.get('sigma'),
                "metadata": d.metadata
            })

        result_entry = {
            'sample_id': i,
            'doc_name': doc_name,
            'doc_link': sample.get('doc_link', ''),
            'question': query,
            'question_type': sample.get('question_type', ''),
            'question_reasoning': sample.get('question_reasoning', ''),
            'reference_answer': sample.get('answer', ''),
            'gold_evidence': gold_str,
            'gold_evidence_segments': gold_segs,  # <--- FIXED: Now included for evaluation
            'generated_answer': ans,
            'generation_length': len(ans),
            'context_length': len(context_text),
            'confidence_margin': margin_conf,
            'confidence_mi': conf_mi,
            'confidence_sigma': conf_sigma,
            'retrieved_chunks': retrieved_chunks,
            'mc_t': experiment.mc_t,
            'mc_alpha': experiment.mc_alpha
        }
        results.append(result_entry)
        
        if i < 2:
            logger.info(f"\n[Sample {i}] Confidence: Margin={margin_conf:.4f}, MI(Neg)={conf_mi:.4f}")
            debug_path = debug_dir / f"sample_{i}_uncertainty.json"
            with open(debug_path, "w") as f:
                json.dump(result_entry, f, indent=2)

    return results

# --- PART F: RISK-COVERAGE ANALYSIS TOOLS ---

def analyze_risk_coverage(results_file: str, output_dir: str):
    """
    Reads a scored results file, calculates risk-coverage curves, and saves CSV/Plot.
    """
    logger.info(f"Analyzing Risk-Coverage for: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to open results file: {e}")
        return
        
    if isinstance(data, dict):
        results = data.get('results', [])
    else:
        results = data
        
    processed_data = []
    for r in results:
        is_correct = 0
        if 'generation_evaluation' in r:
            # Check for LLM Judge correctness
            # The evaluator outputs 'llm_judge_correctness' as boolean or 1/0
            correctness = r['generation_evaluation'].get('llm_judge_correctness')
            if correctness is True or correctness == 1:
                is_correct = 1
            # Fallback to high BERTScore if judge is missing/failed
            elif r['generation_evaluation'].get('bertscore_f1', 0) > 0.85:
                is_correct = 1
        
        processed_data.append({
            'margin': r.get('confidence_margin', -999),
            'mi': r.get('confidence_mi', -999),
            'sigma': r.get('confidence_sigma', -999),
            'correct': is_correct
        })
        
    df = pd.DataFrame(processed_data)
    if df.empty:
        logger.warning("No data found for analysis.")
        return

    methods = {'Margin': 'margin', 'MI': 'mi'}
    coverage_levels = np.linspace(0.1, 1.0, 10)
    analysis_results = []
    
    plt.figure(figsize=(8, 6))
    
    for label, col in methods.items():
        if col not in df.columns: continue
        
        df_sorted = df.sort_values(by=col, ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        
        risks = []
        coverages = []
        
        for cov in coverage_levels:
            k = int(cov * n)
            if k == 0: continue
            
            subset = df_sorted.iloc[:k]
            accuracy = subset['correct'].mean()
            risk = 1.0 - accuracy
            
            risks.append(risk)
            coverages.append(cov)
            
            analysis_results.append({
                'method': label,
                'coverage': cov,
                'risk': risk,
                'accuracy': accuracy,
                'num_answered': k
            })
            
        plt.plot(coverages, risks, marker='o', label=label)

    out_csv = Path(output_dir) / "risk_coverage_data.csv"
    pd.DataFrame(analysis_results).to_csv(out_csv, index=False)
    logger.info(f"Saved Risk-Coverage CSV to {out_csv}")
    
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - Accuracy)")
    plt.title("Risk-Coverage Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_plot = Path(output_dir) / "risk_coverage_plot.png"
    plt.savefig(out_plot)
    logger.info(f"Saved Risk-Coverage Plot to {out_plot}")
    plt.close()