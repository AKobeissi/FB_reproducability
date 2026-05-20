"""
K-Fold Cross-Validation for MG-PEAR Lite (LoRA + Low-Rank Heads)

This script mirrors the cross-validation strategy in train_k_fold2.py:
1. Split documents into k folds
2. For each fold: train on k-1 folds, evaluate on held-out fold
3. Combine predictions from all folds (unbiased estimates for ALL samples)
4. Report aggregated metrics across folds
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer

from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf
from src.evaluation.retrieval_evaluator import RetrievalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class KFoldLoRAConfig:
    # K-Fold settings
    n_folds: int = 5
    random_seed: int = 42

    # Backbone settings
    backbone_name: str = "BAAI/bge-m3"
    max_seq_length: int = 512
    freeze_backbone: bool = True
    use_fp16: bool = True
    gradient_checkpointing: bool = False

    # Low-rank heads
    rank: int = 64

    # LoRA settings
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    lora_strategy: str = "baseline"  # baseline | dual_stage | hierarchical

    # Training settings
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_every_n_epochs: int = 5
    patience: int = 5

    # Loss weights
    lambda_page: float = 0.5
    lambda_chunk: float = 0.5

    # Page/chunk settings
    max_page_chars: int = 2000
    page_k: int = 5
    chunk_k: int = 5
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # Chunk positive selection
    chunk_positive_mode: str = "relaxed"  # strict | relaxed
    chunk_rouge_threshold: float = 0.5
    max_chunks_per_question: int = 8

    # Data settings
    pdf_dir: str = "pdfs"

    # Output settings
    output_dir: str = "results/kfold_mgpear_lite"
    save_models: bool = False

    # Inference settings
    embedding_batch_size: int = 16


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PageRecord:
    doc_name: str
    page_num: int
    page_text: str
    page_id: str = field(init=False)

    def __post_init__(self):
        self.page_id = f"{self.doc_name}_p{self.page_num}"
        self.page_text = self._normalize_text(self.page_text)

    @staticmethod
    def _normalize_text(text: str, max_chars: int = 2000) -> str:
        if not text:
            return " "
        import re
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text[:max_chars] if text else " "


@dataclass
class TrainingMetrics:
    fold: int
    epoch_losses: List[float] = field(default_factory=list)
    eval_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_page_recall: float = 0.0
    training_time_seconds: float = 0.0


# =============================================================================
# Data Loading
# =============================================================================

def load_financebench_data() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    logger.info("Loading FinanceBench data...")
    loader = FinanceBenchLoader()
    df = loader.load_data()
    data = df.to_dict("records")
    logger.info(f"Loaded {len(data)} samples from {len(df['doc_name'].unique())} documents")
    return df, data


def build_page_records(pdf_dir: Path, doc_names: Set[str], max_chars: int) -> Dict[str, List[PageRecord]]:
    pages_by_doc: Dict[str, List[PageRecord]] = {}
    for doc_name in tqdm(doc_names, desc="Loading PDFs"):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        try:
            raw_pages = extract_pages_from_pdf(pdf_path, doc_name)
            page_records = [
                PageRecord(doc_name=doc_name, page_num=p["page"], page_text=p["text"])
                for p in raw_pages
            ]
            pages_by_doc[doc_name] = page_records
        except Exception as exc:
            logger.warning(f"Failed to load {pdf_path}: {exc}")
    total_pages = sum(len(pages) for pages in pages_by_doc.values())
    logger.info(f"Loaded {total_pages} pages from {len(pages_by_doc)} documents")
    return pages_by_doc


def create_document_folds(all_docs: List[str], n_folds: int, seed: int) -> List[Set[str]]:
    random.seed(seed)
    docs_shuffled = all_docs.copy()
    random.shuffle(docs_shuffled)

    fold_size = len(docs_shuffled) // n_folds
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = len(docs_shuffled) if i == n_folds - 1 else start_idx + fold_size
        folds.append(set(docs_shuffled[start_idx:end_idx]))
    return folds


# =============================================================================
# Model
# =============================================================================

def _resolve_lora_targets(backbone: nn.Module, requested: Tuple[str, ...]) -> List[str]:
    name_set = {name.split(".")[-1] for name, _ in backbone.named_modules()}

    if requested and all(name in name_set for name in requested):
        return list(requested)

    candidates = [
        ("q_proj", "v_proj"),
        ("query", "value"),
        ("Wq", "Wv"),
        ("q", "v"),
    ]

    for pair in candidates:
        if all(name in name_set for name in pair):
            logger.warning(
                "LoRA target modules %s not found; using %s instead.",
                requested,
                pair,
            )
            return list(pair)

    raise ValueError(
        "No compatible LoRA target modules found in backbone. "
        "Tried: q_proj/v_proj, query/value, Wq/Wv, q/v. "
        "Set --use-lora off or update lora_target_modules."
    )


def _add_lora_adapter(backbone: nn.Module, adapter_name: str, config: KFoldLoRAConfig) -> None:
    if not hasattr(backbone, "add_adapter"):
        raise RuntimeError("LoRA adapters require a PEFT model with add_adapter support.")
    target_modules = _resolve_lora_targets(backbone, config.lora_target_modules)
    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    backbone.add_adapter(adapter_name, lora_cfg)

class MGPEARLiteModel(nn.Module):
    def __init__(self, config: KFoldLoRAConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name, use_fast=True)
        backbone = AutoModel.from_pretrained(config.backbone_name, trust_remote_code=True)

        enable_checkpointing = config.gradient_checkpointing
        if config.use_lora and config.lora_strategy == "dual_stage" and config.gradient_checkpointing:
            logger.warning("Disabling gradient checkpointing for dual_stage LoRA to avoid checkpoint errors.")
            enable_checkpointing = False

        if enable_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()

        if config.use_lora:
            if get_peft_model is None or LoraConfig is None:
                raise RuntimeError("peft is required for LoRA. Install it with: pip install peft")
            target_modules = _resolve_lora_targets(backbone, config.lora_target_modules)
            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_cfg)

            if config.lora_strategy == "dual_stage":
                _add_lora_adapter(backbone, "query", config)
                if hasattr(backbone, "set_adapter"):
                    backbone.set_adapter("query")
            elif config.lora_strategy == "hierarchical":
                _add_lora_adapter(backbone, "page", config)
                _add_lora_adapter(backbone, "chunk", config)
                if hasattr(backbone, "set_adapter"):
                    backbone.set_adapter("page")

        self.backbone = backbone

        hidden_size = backbone.config.hidden_size
        self.query_page = nn.Linear(hidden_size, config.rank, bias=False)
        self.page_proj = nn.Linear(hidden_size, config.rank, bias=False)
        self.query_chunk = nn.Linear(hidden_size, config.rank, bias=False)
        self.chunk_proj = nn.Linear(hidden_size, config.rank, bias=False)

        if config.freeze_backbone and not config.use_lora:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.to(device)

    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return self.encode_texts_with_adapter(texts, adapter=None, disable_adapter=False)

    def encode_texts_with_adapter(
        self,
        texts: List[str],
        adapter: Optional[str],
        disable_adapter: bool,
    ) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.config.use_lora:
            if disable_adapter and hasattr(self.backbone, "disable_adapter"):
                ctx = self.backbone.disable_adapter()
            else:
                ctx = None
                if adapter and hasattr(self.backbone, "set_adapter"):
                    self.backbone.set_adapter(adapter)
        else:
            ctx = None

        if self.config.use_fp16 and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                if ctx:
                    with ctx:
                        outputs = self.backbone(**inputs)
                else:
                    outputs = self.backbone(**inputs)
        else:
            if ctx:
                with ctx:
                    outputs = self.backbone(**inputs)
            else:
                outputs = self.backbone(**inputs)
        pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def get_adapter_for(self, task: str, role: str) -> Tuple[Optional[str], bool]:
        if not self.config.use_lora:
            return None, False

        if self.config.lora_strategy == "dual_stage":
            if role == "query":
                return "query", False
            return None, True

        if self.config.lora_strategy == "hierarchical":
            if task == "page":
                return "page", False
            return "chunk", False

        return None, False

    def project_page(self, h_q: torch.Tensor, h_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_q = self.query_page(h_q)
        z_p = self.page_proj(h_p)
        return F.normalize(z_q, p=2, dim=1), F.normalize(z_p, p=2, dim=1)

    def project_chunk(self, h_q: torch.Tensor, h_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_q = self.query_chunk(h_q)
        z_c = self.chunk_proj(h_c)
        return F.normalize(z_q, p=2, dim=1), F.normalize(z_c, p=2, dim=1)


# =============================================================================
# Training Data
# =============================================================================

class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.pairs[idx]


def _pair_collate(batch: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    return queries, positives


def _extract_gold_pages(row: pd.Series) -> Set[int]:
    gold_pages: Set[int] = set()
    evidence_list = row.get("evidence", [])
    for ev in evidence_list:
        p = ev.get("evidence_page_num") or ev.get("page_ix") or ev.get("page")
        if p is not None:
            gold_pages.add(int(p))
    return gold_pages


def _extract_evidence_texts(row: pd.Series) -> List[str]:
    evidence_list = row.get("evidence", [])
    texts = []
    for ev in evidence_list:
        text = ev.get("evidence_text") or ev.get("text") or ""
        if text:
            texts.append(text)
    return texts


def create_page_pairs(
    df: pd.DataFrame,
    train_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    examples_per_question: int = 4,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    skipped = 0

    for _, row in df.iterrows():
        doc_name = row["doc_name"]
        if doc_name not in train_docs or doc_name not in pages_by_doc:
            continue

        gold_pages = _extract_gold_pages(row)
        if not gold_pages:
            skipped += 1
            continue

        doc_pages = pages_by_doc[doc_name]
        valid_gold_pages = {p for p in gold_pages if 0 <= p < len(doc_pages)}
        if not valid_gold_pages:
            skipped += 1
            continue

        question = row["question"]
        for _ in range(examples_per_question):
            pos_page = doc_pages[random.choice(list(valid_gold_pages))]
            pairs.append({"query": question, "positive": pos_page.page_text})

    logger.info(f"Created {len(pairs)} page pairs (skipped {skipped})")
    return pairs


def create_chunk_pairs(
    df: pd.DataFrame,
    train_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    text_splitter: RecursiveCharacterTextSplitter,
    config: KFoldLoRAConfig,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    skipped = 0
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for _, row in df.iterrows():
        doc_name = row["doc_name"]
        if doc_name not in train_docs or doc_name not in pages_by_doc:
            continue

        gold_pages = _extract_gold_pages(row)
        if not gold_pages:
            skipped += 1
            continue

        doc_pages = pages_by_doc[doc_name]
        valid_gold_pages = [p for p in gold_pages if 0 <= p < len(doc_pages)]
        if not valid_gold_pages:
            skipped += 1
            continue

        evidence_texts = _extract_evidence_texts(row)
        question = row["question"]

        candidate_chunks: List[str] = []
        for p in valid_gold_pages:
            page_text = doc_pages[p].page_text
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                cleaned = " ".join(chunk.split())
                if cleaned:
                    candidate_chunks.append(cleaned)

        if not candidate_chunks:
            skipped += 1
            continue

        selected_chunks: List[str] = []
        if config.chunk_positive_mode == "relaxed" or not evidence_texts:
            selected_chunks = candidate_chunks
        else:
            for chunk in candidate_chunks:
                max_rouge = 0.0
                for ev_text in evidence_texts:
                    score = rouge.score(ev_text, chunk)["rougeL"].fmeasure
                    if score > max_rouge:
                        max_rouge = score
                if max_rouge >= config.chunk_rouge_threshold:
                    selected_chunks.append(chunk)

        if not selected_chunks:
            skipped += 1
            continue

        if config.max_chunks_per_question > 0:
            random.shuffle(selected_chunks)
            selected_chunks = selected_chunks[: config.max_chunks_per_question]

        for chunk in selected_chunks:
            pairs.append({"query": question, "positive": chunk})

    logger.info(f"Created {len(pairs)} chunk pairs (skipped {skipped})")
    return pairs


# =============================================================================
# Training Utilities
# =============================================================================

def mnr_loss(query_vecs: torch.Tensor, doc_vecs: torch.Tensor) -> torch.Tensor:
    if query_vecs.size(0) < 2:
        return torch.tensor(0.0, device=query_vecs.device)
    logits = torch.matmul(query_vecs, doc_vecs.T)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def train_fold_model(
    fold_idx: int,
    train_docs: Set[str],
    val_docs: Set[str],
    df: pd.DataFrame,
    pages_by_doc: Dict[str, List[PageRecord]],
    config: KFoldLoRAConfig,
    output_dir: Path,
) -> Tuple[MGPEARLiteModel, TrainingMetrics]:
    start_time = time.time()

    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FOLD {fold_idx + 1}/{config.n_folds}")
    logger.info(f"Train docs: {len(train_docs)}, Val docs: {len(val_docs)}")
    logger.info(f"{'='*80}")

    fold_seed = config.random_seed + fold_idx
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGPEARLiteModel(config, device=device)

    tokenizer = model.tokenizer
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )

    page_pairs = create_page_pairs(df, train_docs, pages_by_doc, examples_per_question=4)
    chunk_pairs = create_chunk_pairs(df, train_docs, pages_by_doc, text_splitter, config)

    if not page_pairs or not chunk_pairs:
        raise ValueError(f"Insufficient training data for fold {fold_idx}")

    page_loader = DataLoader(
        PairDataset(page_pairs),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_pair_collate,
    )
    chunk_loader = DataLoader(
        PairDataset(chunk_pairs),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_pair_collate,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    steps_per_epoch = max(len(page_loader), len(chunk_loader))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    def warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    metrics = TrainingMetrics(fold=fold_idx)
    best_model_state = None

    baseline = evaluate_page_retrieval(model, df, val_docs, pages_by_doc, config)
    metrics.eval_metrics.append({"epoch": 0, **baseline})
    logger.info(f"Baseline: {baseline}")

    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        page_iter = iter(page_loader)
        chunk_iter = iter(chunk_loader)

        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            loss_page = torch.tensor(0.0, device=device)
            loss_chunk = torch.tensor(0.0, device=device)

            try:
                page_batch = next(page_iter)
            except StopIteration:
                page_iter = iter(page_loader)
                page_batch = next(page_iter)

            try:
                chunk_batch = next(chunk_iter)
            except StopIteration:
                chunk_iter = iter(chunk_loader)
                chunk_batch = next(chunk_iter)

            page_queries, page_docs = page_batch
            chunk_queries, chunk_docs = chunk_batch

            q_adapter, q_disable = model.get_adapter_for("page", "query")
            d_adapter, d_disable = model.get_adapter_for("page", "doc")
            h_q_page = model.encode_texts_with_adapter(page_queries, q_adapter, q_disable)
            h_p_page = model.encode_texts_with_adapter(page_docs, d_adapter, d_disable)
            z_q_page, z_p_page = model.project_page(h_q_page, h_p_page)
            loss_page = mnr_loss(z_q_page, z_p_page)

            q_adapter, q_disable = model.get_adapter_for("chunk", "query")
            d_adapter, d_disable = model.get_adapter_for("chunk", "doc")
            h_q_chunk = model.encode_texts_with_adapter(chunk_queries, q_adapter, q_disable)
            h_c_chunk = model.encode_texts_with_adapter(chunk_docs, d_adapter, d_disable)
            z_q_chunk, z_c_chunk = model.project_chunk(h_q_chunk, h_c_chunk)
            loss_chunk = mnr_loss(z_q_chunk, z_c_chunk)

            loss = config.lambda_page * loss_page + config.lambda_chunk * loss_chunk
            loss.backward()
            optimizer.step()

            if warmup_steps > 0 and step < warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, steps_per_epoch)
        metrics.epoch_losses.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f}")

        if (epoch + 1) % config.eval_every_n_epochs == 0 or epoch == config.epochs - 1:
            eval_metrics = evaluate_page_retrieval(model, df, val_docs, pages_by_doc, config)
            eval_metrics["epoch"] = epoch + 1
            metrics.eval_metrics.append(eval_metrics)
            logger.info(f"Epoch {epoch + 1}: {eval_metrics}")

            if eval_metrics["page_recall@k"] > metrics.best_page_recall:
                metrics.best_page_recall = eval_metrics["page_recall@k"]
                metrics.best_epoch = epoch + 1
                best_model_state = {
                    "heads": {
                        "query_page": model.query_page.state_dict(),
                        "page_proj": model.page_proj.state_dict(),
                        "query_chunk": model.query_chunk.state_dict(),
                        "chunk_proj": model.chunk_proj.state_dict(),
                    },
                    "backbone": model.backbone.state_dict() if config.use_lora else None,
                }
                epochs_without_improvement = 0
                logger.info(f"New best model (page_recall@k: {metrics.best_page_recall:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    if best_model_state:
        model.query_page.load_state_dict(best_model_state["heads"]["query_page"])
        model.page_proj.load_state_dict(best_model_state["heads"]["page_proj"])
        model.query_chunk.load_state_dict(best_model_state["heads"]["query_chunk"])
        model.chunk_proj.load_state_dict(best_model_state["heads"]["chunk_proj"])
        if config.use_lora and best_model_state["backbone"] is not None:
            model.backbone.load_state_dict(best_model_state["backbone"], strict=False)

    if config.save_models:
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save learned heads (tiny ~10-50MB)
        torch.save(
            {
                "config": asdict(config),
                "base_model": config.backbone_name,  # Store base model name for reloading
                "heads": {
                    "query_page": model.query_page.state_dict(),
                    "page_proj": model.page_proj.state_dict(),
                    "query_chunk": model.query_chunk.state_dict(),
                    "chunk_proj": model.chunk_proj.state_dict(),
                },
            },
            fold_dir / "mgpear_heads.pt",
        )
        
        # Save only LoRA adapters if used (small ~few MB vs 2.2GB base model)
        if config.use_lora and hasattr(model.backbone, "save_pretrained"):
            if hasattr(model.backbone, "peft_config"):
                for adapter_name in model.backbone.peft_config.keys():
                    if hasattr(model.backbone, "set_adapter"):
                        model.backbone.set_adapter(adapter_name)
                    # This only saves the LoRA weights, not the full model
                    model.backbone.save_pretrained(fold_dir / f"lora_{adapter_name}")
            else:
                model.backbone.save_pretrained(fold_dir / "lora_backbone")
        
        # Save tokenizer config (tiny)
        model.tokenizer.save_pretrained(fold_dir / "tokenizer")
        
        logger.info(f"Saved lightweight model components to {fold_dir}")
        logger.info(f"  - Learned heads: {fold_dir / 'mgpear_heads.pt'}")
        if config.use_lora:
            logger.info(f"  - LoRA adapters only (base model reloaded from HuggingFace)")

    metrics.training_time_seconds = time.time() - start_time
    logger.info(
        f"Fold {fold_idx + 1} training complete in {metrics.training_time_seconds:.1f}s "
        f"(best page_recall@k={metrics.best_page_recall:.4f} at epoch {metrics.best_epoch})"
    )

    return model, metrics


# =============================================================================
# Evaluation and Inference
# =============================================================================

@torch.no_grad()
def evaluate_page_retrieval(
    model: MGPEARLiteModel,
    df: pd.DataFrame,
    eval_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    config: KFoldLoRAConfig,
) -> Dict[str, float]:
    model.eval()
    page_hits = 0
    page_recalls = []
    total_questions = 0

    for _, row in df.iterrows():
        doc_name = row["doc_name"]
        if doc_name not in eval_docs or doc_name not in pages_by_doc:
            continue

        gold_pages = _extract_gold_pages(row)
        if not gold_pages:
            continue

        doc_pages = pages_by_doc[doc_name]
        valid_gold_pages = {p for p in gold_pages if 0 <= p < len(doc_pages)}
        if not valid_gold_pages:
            continue

        total_questions += 1
        question = row["question"]

        q_adapter, q_disable = model.get_adapter_for("page", "query")
        d_adapter, d_disable = model.get_adapter_for("page", "doc")
        h_q = model.encode_texts_with_adapter([question], q_adapter, q_disable)
        h_p = model.encode_texts_with_adapter([p.page_text for p in doc_pages], d_adapter, d_disable)
        z_q, z_p = model.project_page(h_q, h_p)

        scores = torch.matmul(z_q, z_p.T).squeeze(0)
        top_indices = torch.argsort(scores, descending=True)[: config.page_k].cpu().tolist()

        retrieved_pages = set(top_indices)
        hits = retrieved_pages & valid_gold_pages
        if hits:
            page_hits += 1
        page_recalls.append(len(hits) / len(valid_gold_pages))

    if total_questions == 0:
        return {"page_hit@k": 0.0, "page_recall@k": 0.0, "n_questions": 0}

    return {
        "page_hit@k": page_hits / total_questions,
        "page_recall@k": float(np.mean(page_recalls)) if page_recalls else 0.0,
        "n_questions": total_questions,
    }


@torch.no_grad()
def run_page_then_chunk_inference(
    model: MGPEARLiteModel,
    samples: List[Dict[str, Any]],
    pages_by_doc: Dict[str, List[PageRecord]],
    config: KFoldLoRAConfig,
) -> List[Dict[str, Any]]:
    model.eval()
    tokenizer = model.tokenizer
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )

    all_pages: List[PageRecord] = []
    for _, pages in pages_by_doc.items():
        all_pages.extend(pages)

    all_page_texts = [p.page_text for p in all_pages]
    all_page_meta = [(p.doc_name, p.page_num) for p in all_pages]

    page_embeddings = []
    for i in range(0, len(all_page_texts), config.embedding_batch_size):
        batch = all_page_texts[i : i + config.embedding_batch_size]
        d_adapter, d_disable = model.get_adapter_for("page", "doc")
        h_p = model.encode_texts_with_adapter(batch, d_adapter, d_disable)
        _, z_p = model.project_page(h_p, h_p)
        page_embeddings.append(z_p.cpu())
    page_embeddings = torch.cat(page_embeddings, dim=0)

    chunk_texts: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []
    page_to_chunk_indices: Dict[Tuple[str, int], List[int]] = {}

    for doc_name, pages in pages_by_doc.items():
        for page in pages:
            chunks = text_splitter.split_text(page.page_text)
            for chunk in chunks:
                cleaned = " ".join(chunk.split())
                if not cleaned:
                    continue
                idx = len(chunk_texts)
                chunk_texts.append(cleaned)
                chunk_metas.append({"doc_name": doc_name, "page": page.page_num})
                page_to_chunk_indices.setdefault((doc_name, page.page_num), []).append(idx)

    chunk_embeddings = []
    for i in range(0, len(chunk_texts), config.embedding_batch_size):
        batch = chunk_texts[i : i + config.embedding_batch_size]
        d_adapter, d_disable = model.get_adapter_for("chunk", "doc")
        h_c = model.encode_texts_with_adapter(batch, d_adapter, d_disable)
        _, z_c = model.project_chunk(h_c, h_c)
        chunk_embeddings.append(z_c.cpu())
    chunk_embeddings = torch.cat(chunk_embeddings, dim=0) if chunk_embeddings else torch.empty((0, config.rank))

    results = []
    for sample in tqdm(samples, desc="Inference"):
        question = sample["question"]
        q_adapter, q_disable = model.get_adapter_for("page", "query")
        h_q = model.encode_texts_with_adapter([question], q_adapter, q_disable)
        z_q_page, _ = model.project_page(h_q, h_q)
        q_adapter, q_disable = model.get_adapter_for("chunk", "query")
        h_q_chunk = model.encode_texts_with_adapter([question], q_adapter, q_disable)
        z_q_chunk, _ = model.project_chunk(h_q_chunk, h_q_chunk)

        page_scores = torch.matmul(z_q_page.cpu(), page_embeddings.T).squeeze(0)
        top_page_indices = torch.argsort(page_scores, descending=True)[: config.page_k].tolist()

        retrieved_pages_meta = [all_page_meta[idx] for idx in top_page_indices]
        retrieved_pages = [f"{doc}_p{page}" for doc, page in retrieved_pages_meta]

        candidate_chunk_indices: List[int] = []
        for doc, page in retrieved_pages_meta:
            candidate_chunk_indices.extend(page_to_chunk_indices.get((doc, page), []))

        retrieved_chunks = []
        if candidate_chunk_indices:
            candidate_embs = chunk_embeddings[candidate_chunk_indices]
            chunk_scores = torch.matmul(z_q_chunk.cpu(), candidate_embs.T).squeeze(0)
            top_chunk_local = torch.argsort(chunk_scores, descending=True)[: config.chunk_k].tolist()
            for local_idx in top_chunk_local:
                global_idx = candidate_chunk_indices[local_idx]
                retrieved_chunks.append(
                    {
                        "text": chunk_texts[global_idx],
                        "metadata": chunk_metas[global_idx],
                        "score": float(chunk_scores[local_idx]),
                    }
                )

        evidence_list = sample.get("evidence", [])
        if not isinstance(evidence_list, list):
            try:
                evidence_list = list(evidence_list)
            except Exception:
                evidence_list = []
        gold_segments = []
        for ev in evidence_list:
            gold_segments.append(
                {
                    "doc_name": ev.get("doc_name", sample.get("doc_name")),
                    "page": ev.get("evidence_page_num") or ev.get("page_ix") or ev.get("page"),
                    "text": ev.get("evidence_text", ""),
                }
            )

        result = {
            **sample,
            "retrieved_pages": retrieved_pages,
            "retrieved_chunks": retrieved_chunks,
            "gold_evidence_segments": gold_segments,
            "gold_evidence": "\n\n".join([s.get("text", "") for s in gold_segments]),
            "reference_answer": sample.get("answer", ""),
            "model_answer": "Generation skipped",
        }
        results.append(result)

    return results


def compute_full_metrics(results: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    evaluator = RetrievalEvaluator()
    retrieval_metrics = evaluator.compute_metrics(results, k_values=k_values)
    return retrieval_metrics


# =============================================================================
# K-Fold Orchestration
# =============================================================================

def run_kfold_cross_validation(config: KFoldLoRAConfig) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("K-FOLD CV (MG-PEAR LITE + LoRA)")
    logger.info(f"LoRA strategy: {config.lora_strategy}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    df, data = load_financebench_data()
    all_docs = list(df["doc_name"].unique())
    folds = create_document_folds(all_docs, config.n_folds, config.random_seed)

    logger.info(f"\nDocument distribution across {config.n_folds} folds:")
    for i, fold in enumerate(folds):
        logger.info(f"  Fold {i + 1}: {len(fold)} documents")

    fold_assignments = {f"fold_{i}": list(fold) for i, fold in enumerate(folds)}
    with open(output_dir / "fold_assignments.json", "w") as f:
        json.dump(fold_assignments, f, indent=2)

    pdf_dir = Path(config.pdf_dir)
    pages_by_doc = build_page_records(pdf_dir, set(all_docs), config.max_page_chars)

    all_fold_metrics: List[TrainingMetrics] = []
    all_predictions: List[Dict[str, Any]] = []
    fold_eval_results: List[Dict[str, Any]] = []

    for fold_idx in range(config.n_folds):
        fold_output_dir = output_dir / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        test_docs = folds[fold_idx]
        train_docs = set().union(*[f for i, f in enumerate(folds) if i != fold_idx])
        test_samples = [s for s in data if s["doc_name"] in test_docs]

        logger.info(
            f"\nFold {fold_idx + 1}: Train={len(train_docs)} docs, Test={len(test_docs)} docs, "
            f"Test samples={len(test_samples)}"
        )

        model, training_metrics = train_fold_model(
            fold_idx=fold_idx,
            train_docs=train_docs,
            val_docs=test_docs,
            df=df,
            pages_by_doc=pages_by_doc,
            config=config,
            output_dir=fold_output_dir,
        )
        all_fold_metrics.append(training_metrics)

        with open(fold_output_dir / "training_metrics.json", "w") as f:
            json.dump(
                {
                    "fold": fold_idx,
                    "best_epoch": training_metrics.best_epoch,
                    "best_page_recall": training_metrics.best_page_recall,
                    "training_time_seconds": training_metrics.training_time_seconds,
                    "eval_metrics": training_metrics.eval_metrics,
                },
                f,
                indent=2,
            )

        logger.info(f"\nRunning inference on fold {fold_idx + 1} test set ({len(test_samples)} samples)...")
        fold_predictions = run_page_then_chunk_inference(model, test_samples, pages_by_doc, config)
        fold_metrics = compute_full_metrics(fold_predictions)

        fold_eval_results.append({"fold": fold_idx, "n_samples": len(test_samples), "metrics": fold_metrics})
        logger.info(f"Fold {fold_idx + 1} metrics: {fold_metrics}")

        with open(fold_output_dir / "predictions.json", "w") as f:
            json.dump(fold_predictions, f, indent=2, default=str)

        all_predictions.extend(fold_predictions)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING RESULTS ACROSS ALL FOLDS")
    logger.info("=" * 80)

    all_metrics = compute_full_metrics(all_predictions)

    metric_keys = set()
    for fold_result in fold_eval_results:
        metric_keys.update(fold_result["metrics"].keys())

    aggregated_stats = {}
    for key in metric_keys:
        values = [f["metrics"].get(key, 0) for f in fold_eval_results]
        aggregated_stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    training_stats = {
        "best_page_recalls": [m.best_page_recall for m in all_fold_metrics],
        "best_epochs": [m.best_epoch for m in all_fold_metrics],
        "training_times": [m.training_time_seconds for m in all_fold_metrics],
        "mean_best_page_recall": float(np.mean([m.best_page_recall for m in all_fold_metrics])),
        "std_best_page_recall": float(np.std([m.best_page_recall for m in all_fold_metrics])),
        "total_training_time": sum(m.training_time_seconds for m in all_fold_metrics),
    }

    with open(output_dir / "all_predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)

    final_results = {
        "config": asdict(config),
        "n_folds": config.n_folds,
        "n_total_samples": len(all_predictions),
        "timestamp": timestamp,
        "overall_metrics": all_metrics,
        "per_fold_metrics": fold_eval_results,
        "aggregated_statistics": aggregated_stats,
        "training_statistics": training_stats,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    summary_lines = [
        "=" * 80,
        f"K-FOLD MG-PEAR LITE RESULTS ({config.n_folds} folds)",
        "=" * 80,
        f"Total samples: {len(all_predictions)}",
        f"Total training time: {training_stats['total_training_time']:.1f}s",
        "",
        "RETRIEVAL METRICS (mean +/- std across folds):",
        "-" * 40,
    ]

    for key in ["doc_hit@5", "page_hit@5", "chunk_hit@5", "doc_recall@5", "page_recall@5", "chunk_recall@5", "mrr"]:
        if key in aggregated_stats:
            stats = aggregated_stats[key]
            summary_lines.append(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    summary_lines.extend(
        [
            "",
            "TRAINING STATISTICS:",
            "-" * 40,
            f"  Best page_recall@k: {training_stats['mean_best_page_recall']:.4f} +/- {training_stats['std_best_page_recall']:.4f}",
            f"  Best epochs: {training_stats['best_epochs']}",
            "",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    logger.info(f"\nAll results saved to: {output_dir}")
    return final_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-Fold CV for MG-PEAR Lite (LoRA + Low-Rank Heads)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--page-k", type=int, default=5)
    parser.add_argument("--chunk-k", type=int, default=5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument(
        "--lora-strategy",
        type=str,
        default="baseline",
        choices=["baseline", "dual_stage", "hierarchical"],
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/kfold_mgpear_lite")
    parser.add_argument("--pdf-dir", type=str, default="pdfs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--save-models", action="store_true", help="Save per-fold lightweight checkpoints")
    parser.add_argument("--chunk-positive-mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--chunk-rouge-threshold", type=float, default=0.5)

    args = parser.parse_args()

    cfg = KFoldLoRAConfig(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        page_k=args.page_k,
        chunk_k=args.chunk_k,
        rank=args.rank,
        lora_strategy=args.lora_strategy,
        max_seq_length=args.max_seq_length,
        embedding_batch_size=args.embedding_batch_size,
        use_fp16=not args.no_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        random_seed=args.seed,
        use_lora=args.use_lora,
        save_models=args.save_models,
        chunk_positive_mode=args.chunk_positive_mode,
        chunk_rouge_threshold=args.chunk_rouge_threshold,
    )

    run_kfold_cross_validation(cfg)
