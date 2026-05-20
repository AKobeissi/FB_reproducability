"""
Train standard cross-encoder with cross-entropy loss.
Saves checkpoint, metrics, predictions per epoch.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.models.cross_encoder import CrossEncoder
from src.training.dataset import PairDataset
from src.utils.seed import set_seed
from src.utils.logging import get_logger, timestamped_run_dir
from src.utils.io import save_json


def train(args) -> None:
    set_seed(args.seed)
    logger = get_logger("train_ce")
    run_dir = timestamped_run_dir(args.run_dir, "ce")
    log_file = run_dir / "train.log"
    logger = get_logger("train_ce", log_file=log_file)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Save config
    save_json(vars(args), run_dir / "config.json")

    # Data
    train_ds = PairDataset(args.train_pairs, args.backbone, args.max_length, args.use_metadata)
    val_ds = PairDataset(args.val_pairs, args.backbone, args.max_length, args.use_metadata)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    # Model
    model = CrossEncoder(args.backbone)
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    best_val_loss = float("inf")
    metrics_history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tt_ids = batch["token_type_ids"].to(device) if batch["token_type_ids"].any() else None
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(input_ids, attn, tt_ids)
                loss = criterion(logits, labels) / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": f"{train_loss/(step+1):.4f}"})

        # Validation
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        avg_train_loss = train_loss / len(train_loader)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        metrics_history.append(epoch_metrics)
        logger.info(f"Epoch {epoch+1}: {epoch_metrics}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(run_dir / "best_model")
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")

    save_json(metrics_history, run_dir / "metrics.json")
    logger.info(f"Training done. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Results in: {run_dir}")


def _evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tt_ids = batch["token_type_ids"].to(device) if batch["token_type_ids"].any() else None
            labels = batch["label"].to(device)
            logits = model(input_ids, attn, tt_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", default="data/processed/pairs/finqa_train_pairs.jsonl")
    parser.add_argument("--val_pairs", default="data/processed/pairs/finqa_val_pairs.jsonl")
    parser.add_argument("--backbone", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_metadata", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", default="results/runs")
    args = parser.parse_args()
    train(args)
