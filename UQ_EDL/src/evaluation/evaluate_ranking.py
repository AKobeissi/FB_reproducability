"""
Score a model (CE or EDL) on a set of candidate pairs and compute ranking metrics.
Works for any model type by loading the appropriate class.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import PairDataset
from src.evaluation.metrics import compute_ranking_metrics
from src.utils.io import load_jsonl, save_json, save_jsonl
from src.utils.logging import get_logger


def score_pairs(
    model,
    pairs_path: str,
    backbone: str,
    max_length: int = 512,
    batch_size: int = 16,
    device: torch.device | None = None,
    use_metadata: bool = True,
    model_type: str = "ce",
    beta: float = 0.25,
) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = PairDataset(pairs_path, backbone, max_length, use_metadata)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pairs_raw = load_jsonl(pairs_path)

    all_scores = []
    all_p_relevant = []
    all_uncertainty = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tt_ids = batch["token_type_ids"].to(device) if batch["token_type_ids"].any() else None

            if model_type == "edl":
                outputs = model(input_ids, attn, tt_ids)
                p_rel = outputs["p_relevant"].cpu().numpy()
                unc = outputs["uncertainty"].cpu().numpy()
                score = p_rel - beta * unc
                all_p_relevant.extend(p_rel.tolist())
                all_uncertainty.extend(unc.tolist())
                all_scores.extend(score.tolist())
            else:
                logits = model(input_ids, attn, tt_ids)
                p_rel = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_scores.extend(p_rel.tolist())
                all_p_relevant.extend(p_rel.tolist())
                all_uncertainty.extend([float("nan")] * len(p_rel))

    # Attach scores to raw pairs
    results = []
    for i, pair in enumerate(pairs_raw):
        rec = dict(pair)
        rec["score"] = float(all_scores[i])
        rec["p_relevant"] = float(all_p_relevant[i])
        rec["uncertainty"] = float(all_uncertainty[i])
        results.append(rec)
    return results


def evaluate_ranking_from_scores(
    scored_pairs: list[dict],
    ks: list[int] = [1, 3, 5, 10],
    mrr_k: int = 10,
    ndcg_ks: list[int] = [5, 10],
    map_k: int = 10,
) -> dict:
    # Group by qid and sort by score desc
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in scored_pairs:
        by_qid[p["qid"]].append(p)

    qid_to_ranked: dict[str, list[str]] = {}
    qid_to_gold: dict[str, set[str]] = {}
    for qid, pairs in by_qid.items():
        pairs.sort(key=lambda x: x["score"], reverse=True)
        qid_to_ranked[qid] = [p["candidate_id"] for p in pairs]
        qid_to_gold[qid] = {p["candidate_id"] for p in pairs if p["label"] == 1}

    metrics = compute_ranking_metrics(qid_to_ranked, qid_to_gold, ks, mrr_k, ndcg_ks, map_k)
    return metrics, qid_to_ranked, qid_to_gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--backbone", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model_type", choices=["ce", "edl"], default="ce")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--tag", default="eval")
    args = parser.parse_args()

    logger = get_logger("eval_ranking")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "edl":
        from src.models.evidential_cross_encoder import EvidentialCrossEncoder
        model = EvidentialCrossEncoder.load(args.model_dir)
    else:
        from src.models.cross_encoder import CrossEncoder
        model = CrossEncoder.load(args.model_dir)

    scored = score_pairs(model, args.pairs, args.backbone, args.max_length, args.batch_size, device, model_type=args.model_type, beta=args.beta)
    save_jsonl(scored, out_dir / f"{args.tag}_scored_pairs.jsonl")

    metrics, qid_to_ranked, qid_to_gold = evaluate_ranking_from_scores(scored)
    logger.info(f"Ranking metrics: {metrics}")
    save_json(metrics, out_dir / f"{args.tag}_ranking_metrics.json")
    logger.info(f"Saved to {out_dir}")
