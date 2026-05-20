#!/usr/bin/env python3
"""
run_positional_bias_attn.py
===========================
Demonstrates lost-in-the-middle / positional bias in Qwen2.5-14B-Instruct
for FinanceBench example financebench_id_04735 (ADOBE_2015_10K).

Strategy
--------
Build three multi-chunk prompts where the two gold-evidence chunks
(balance sheet page 58 + cash-flow statement page 62) are placed at:
  BEGINNING  — positions  1–2  out of 20
  MIDDLE     — positions 10–11 out of 20
  END        — positions 19–20 out of 20

The remaining 18 slots are filled with distractor chunks retrieved by a
dense BGE-M3 baseline (all cash-flow or notes chunks, no balance sheet).

For each scenario we:
  1. Generate a full answer (max_new_tokens=350, greedy).
  2. Run a memory-efficient attention capture:
       a. Prefill the prompt with use_cache=True, output_attentions=False
          → no O(seq²) memory pressure.
       b. Feed the first generated token back in with output_attentions=True
          → attention for that single token over the full KV cache:
          shape (48 layers × 40 heads × 1 × input_len) ≈ 27 MB total.
  3. Average over layers and heads → per-input-token importance scores.
  4. Aggregate per chunk.

Outputs (saved to analysis/positional_bias/):
  attention_per_token_{scenario}.png    — token-level attention with chunk bands
  attention_per_chunk.png               — grouped bar chart (3 scenarios × 20 chunks)
  ucurve.png                            — gold-evidence attention vs. gold position
  results.json                          — generated answers + numeric metrics
"""

import json
import os
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE    = PROJECT_ROOT / "data/financebench_open_source.jsonl"
ORACLE_FILE  = PROJECT_ROOT / "outputs/llm_scaling_meta_filter_rr/oracle_predictions/Qwen2.5-14B_oracle.json"
DENSE_FILE   = PROJECT_ROOT / "outputs/gen_comparison_dense_bge_m3/predictions/Qwen2.5-7B.json"
OUT_DIR      = PROJECT_ROOT / "analysis/positional_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ID    = "financebench_id_04735"
MODEL_NAME   = "Qwen/Qwen2.5-14B-Instruct"

MAX_CONTEXT_TOKENS = 3600
MAX_NEW_TOKENS     = 350
N_TOTAL_CHUNKS     = 20   # total chunks in each scenario

GEN_PROMPT = (
    "You are a financial analyst answering questions based on SEC filings. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "say 'I cannot determine this from the provided information.' "
    "Be concise and precise, especially for numerical answers.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

# ============================================================
# 1.  Load data
# ============================================================

print("Loading data…")
with open(ORACLE_FILE) as f:
    oracle_data = {item["financebench_id"]: item for item in json.load(f)}

with open(DENSE_FILE) as f:
    dense_data = {item["financebench_id"]: item for item in json.load(f)}

sample_oracle = oracle_data[SAMPLE_ID]
sample_dense  = dense_data[SAMPLE_ID]

question = sample_oracle["question"]
reference_answer = sample_oracle["reference_answer"]
gold_segments = sample_oracle["gold_evidence_segments"]

# Gold chunks: balance sheet (page 58) and cash-flow statement (page 62)
gold_balance_sheet = gold_segments[0]["text"]   # total current liabilities
gold_cash_flow     = gold_segments[1]["text"]   # operating cash flows

# Distractor chunks from dense retrieval (all cash-flow / notes, NO balance sheet)
distractor_chunks = [c["text"] for c in sample_dense["retrieved_chunks"]]
# Ensure we have enough distractors
while len(distractor_chunks) < N_TOTAL_CHUNKS - 2:
    distractor_chunks.extend(distractor_chunks)
distractor_chunks = distractor_chunks[: N_TOTAL_CHUNKS - 2]

print(f"  Gold chunks: 2  |  Distractor chunks: {len(distractor_chunks)}")

# ============================================================
# 2.  Build three chunk orderings
# ============================================================

gold_pair = [gold_balance_sheet, gold_cash_flow]  # both needed

n_before_mid = (N_TOTAL_CHUNKS - 2) // 2   # 9
n_after_mid  = N_TOTAL_CHUNKS - 2 - n_before_mid  # 9

scenarios = {
    "beginning": gold_pair + distractor_chunks[:N_TOTAL_CHUNKS - 2],
    "middle":    distractor_chunks[:n_before_mid] + gold_pair + distractor_chunks[n_before_mid:n_before_mid + n_after_mid],
    "end":       distractor_chunks[:N_TOTAL_CHUNKS - 2] + gold_pair,
}

# gold chunk indices (0-based) in each scenario
gold_indices = {
    "beginning": (0, 1),
    "middle":    (n_before_mid, n_before_mid + 1),
    "end":       (N_TOTAL_CHUNKS - 2, N_TOTAL_CHUNKS - 1),
}

# ============================================================
# 3.  Load model
# ============================================================

print(f"\nLoading {MODEL_NAME} (4-bit)…")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
).eval()

num_layers = model.config.num_hidden_layers
print(f"  Model loaded  |  layers={num_layers}  |  device_map=auto")

# ============================================================
# Helper: build context and tokenize
# ============================================================

def build_context(chunks, tokenizer, budget):
    """Join chunks with '\n\n', respecting token budget."""
    parts = []
    remaining = budget
    for chunk in chunks:
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        if len(ids) <= remaining:
            parts.append(chunk)
            remaining -= len(ids)
        elif remaining >= 50:
            parts.append(tokenizer.decode(ids[:remaining], skip_special_tokens=True))
            remaining = 0
            break
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def make_inputs(chunks, tokenizer, model_device):
    context = build_context(chunks, tokenizer, MAX_CONTEXT_TOKENS)
    prompt  = GEN_PROMPT.format(context=context, question=question)
    if hasattr(tokenizer, "apply_chat_template"):
        msgs      = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs    = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model_device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model_device)
    return inputs, context


def get_chunk_token_spans(chunks_used, prompt_text, tokenizer):
    """
    For each chunk that was included in the prompt, find the token span
    [start, end) in the tokenised prompt.
    Returns list of (start_tok, end_tok, chunk_idx_in_scenario).
    """
    spans = []
    search_start = 0
    # locate 'Context:\n' header
    ctx_header = "Context:\n"
    ctx_pos = prompt_text.find(ctx_header)
    if ctx_pos == -1:
        ctx_pos = 0
    else:
        ctx_pos += len(ctx_header)

    for ci, chunk in enumerate(chunks_used):
        # truncated chunk might differ — search for first 80 chars
        needle = chunk[:80].strip()
        pos = prompt_text.find(needle, search_start)
        if pos == -1:
            continue
        chunk_end = pos + len(chunk)

        # token indices: count tokens in prefix
        prefix     = prompt_text[:pos]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        chunk_ids  = tokenizer.encode(chunk,  add_special_tokens=False)
        t_start    = len(prefix_ids)
        t_end      = t_start + len(chunk_ids)
        spans.append((t_start, t_end, ci))
        search_start = pos + 1
    return spans


# ============================================================
# 4.  Run generation + attention capture per scenario
# ============================================================

results = {}

for scenario_name, chunks in scenarios.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name.upper()}")
    gold_pos = gold_indices[scenario_name]
    print(f"  Gold chunks at positions: {gold_pos[0]+1} and {gold_pos[1]+1} of {N_TOTAL_CHUNKS}")

    inputs, context_text = make_inputs(chunks, tokenizer, model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"  Prompt token length: {input_len}")

    # --- 4a. Full generation (greedy, no attention overhead) ---
    print("  Generating full answer…")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            do_sample=False,
            output_attentions=False,
        )
    generated_answer = tokenizer.decode(
        gen_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    print(f"  Generated: {generated_answer[:150]}")

    # --- 4b. Attention capture: prefill → KV cache, then decode 1 token ---
    print("  Capturing attention (prefill + single decode step)…")
    with torch.no_grad():
        prefill_out = model(
            **inputs,
            use_cache=True,
            output_attentions=False,
        )
        first_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # One decode step with full attention output
        decode_out = model(
            first_token_id,
            past_key_values=prefill_out.past_key_values,
            output_attentions=True,
            use_cache=False,
        )
        # decode_out.attentions: tuple[num_layers], each (1, n_heads, 1, input_len+1)
        layer_attns = []
        for la in decode_out.attentions:
            # la: (1, n_heads, 1, kv_len)  where kv_len = input_len + 1
            # The last position corresponds to the new token itself; take input_len positions
            a = la[0, :, 0, :input_len].cpu().float().numpy()  # (n_heads, input_len)
            layer_attns.append(a.mean(axis=0))                 # (input_len,)
        mean_attn = np.mean(layer_attns, axis=0)               # (input_len,)

    # --- 4c. Map attention to chunks via token spans ---
    # Reconstruct the full prompt text for span detection
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": GEN_PROMPT.format(context=context_text, question=question)}]
        full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = GEN_PROMPT.format(context=context_text, question=question)

    spans = get_chunk_token_spans(chunks, full_prompt, tokenizer)

    # Aggregate attention per chunk
    chunk_attn = np.zeros(N_TOTAL_CHUNKS)
    for (t_start, t_end, ci) in spans:
        t_start = min(t_start, input_len)
        t_end   = min(t_end,   input_len)
        if t_end > t_start:
            chunk_attn[ci] = float(mean_attn[t_start:t_end].sum())

    # Numeric match: does generated answer contain "0.66"?
    pred_nums = re.findall(r"\b\d+\.?\d*\b", generated_answer.replace(",", ""))
    numeric_match = any(abs(float(n) - 0.66) < 0.02 for n in pred_nums if float(n) < 10)

    results[scenario_name] = {
        "generated_answer": generated_answer,
        "numeric_match":    numeric_match,
        "mean_attn":        mean_attn,          # (input_len,)
        "chunk_attn":       chunk_attn,         # (N_TOTAL_CHUNKS,)
        "spans":            spans,
        "gold_positions":   gold_pos,
        "input_len":        input_len,
    }
    print(f"  Numeric match: {numeric_match}")

    del prefill_out, decode_out
    torch.cuda.empty_cache()

# ============================================================
# 5.  Save results JSON
# ============================================================

out_json = {
    k: {
        "generated_answer": v["generated_answer"],
        "numeric_match":    v["numeric_match"],
        "gold_positions":   list(v["gold_positions"]),
        "chunk_attn":       v["chunk_attn"].tolist(),
    }
    for k, v in results.items()
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(out_json, f, indent=2)
print("\nResults saved →", OUT_DIR / "results.json")

# ============================================================
# 6.  Visualisations
# ============================================================

SCENARIO_COLORS = {
    "beginning": "#2196F3",  # blue
    "middle":    "#FF5722",  # red-orange
    "end":       "#4CAF50",  # green
}
GOLD_COLOR = "#FFD700"

# ---- 6a. Per-token attention (one plot per scenario) --------------------
for scenario_name, res in results.items():
    mean_attn = res["mean_attn"]
    spans     = res["spans"]
    gold_pos  = res["gold_positions"]
    input_len = res["input_len"]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(mean_attn, linewidth=0.6, color=SCENARIO_COLORS[scenario_name], alpha=0.8)

    # Shade chunk bands alternating light/dark, highlight gold chunks
    colors_band = ["#f0f4ff", "#dce8ff"]
    for (t_start, t_end, ci) in spans:
        t_start = min(t_start, input_len)
        t_end   = min(t_end,   input_len)
        if t_end <= t_start:
            continue
        is_gold = ci in gold_pos
        fc = GOLD_COLOR if is_gold else colors_band[ci % 2]
        alpha = 0.55 if is_gold else 0.25
        ax.axvspan(t_start, t_end, facecolor=fc, alpha=alpha, zorder=0)
        ax.text(
            (t_start + t_end) / 2, ax.get_ylim()[1] * 0.95,
            f"C{ci+1}{'*' if is_gold else ''}",
            ha="center", va="top", fontsize=5.5,
            color="#b00000" if is_gold else "#555",
        )

    ax.set_xlabel("Input token position", fontsize=11)
    ax.set_ylabel("Attention weight (layer-head mean)", fontsize=11)
    gp_str = f"{gold_pos[0]+1}–{gold_pos[1]+1}"
    title = (
        f"First-token attention – gold evidence at position {gp_str}/20  "
        f"({'correct' if res['numeric_match'] else 'WRONG'})\n"
        f"Answer: {res['generated_answer'][:80]}…"
    )
    ax.set_title(title, fontsize=10)
    gold_patch  = mpatches.Patch(color=GOLD_COLOR,    alpha=0.6, label=f"Gold evidence (pos {gp_str})")
    other_patch = mpatches.Patch(color="#dce8ff",     alpha=0.6, label="Distractor chunks")
    ax.legend(handles=[gold_patch, other_patch], fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / f"attention_per_token_{scenario_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---- 6b. Per-chunk grouped bar chart ----------------------------------
fig, ax = plt.subplots(figsize=(16, 5))
chunk_positions = np.arange(N_TOTAL_CHUNKS)
bar_width = 0.26
offsets = {"beginning": -bar_width, "middle": 0, "end": bar_width}

for scenario_name, res in results.items():
    chunk_attn = res["chunk_attn"]
    offset     = offsets[scenario_name]
    gold_pos   = res["gold_positions"]

    bars = ax.bar(
        chunk_positions + offset, chunk_attn,
        width=bar_width * 0.92,
        color=SCENARIO_COLORS[scenario_name],
        alpha=0.8,
        label=f"Gold at {scenario_name} (pos {gold_pos[0]+1}–{gold_pos[1]+1})"
               + (" ✓" if res["numeric_match"] else " ✗"),
    )
    # Outline gold bars
    for gi in gold_pos:
        bars[gi].set_edgecolor("#b00000")
        bars[gi].set_linewidth(2)

ax.set_xticks(chunk_positions)
ax.set_xticklabels([f"C{i+1}" for i in range(N_TOTAL_CHUNKS)], fontsize=8)
ax.set_xlabel("Chunk position in context", fontsize=12)
ax.set_ylabel("Summed attention to chunk", fontsize=12)
ax.set_title(
    f"First-token attention per chunk — three positional scenarios\n"
    f"FinanceBench ID {SAMPLE_ID}  (ADOBE 2015 10-K)\n"
    f"Gold chunks (balance sheet + cash-flow) shown with red outline",
    fontsize=11,
)
ax.legend(fontsize=10)
plt.tight_layout()
out_path = OUT_DIR / "attention_per_chunk.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

# ---- 6c. U-curve: gold attention vs. gold position --------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: attention to gold evidence vs. position
gold_attn_vals = []
gold_pos_vals  = []
correct_vals   = []
for scenario_name, res in results.items():
    gp = res["gold_positions"]
    gold_attn = float(res["chunk_attn"][list(gp)].sum())
    gold_center_pos = (gp[0] + gp[1]) / 2 + 1  # 1-indexed centre
    gold_attn_vals.append(gold_attn)
    gold_pos_vals.append(gold_center_pos)
    correct_vals.append(res["numeric_match"])

ax = axes[0]
for x, y, c, s in zip(gold_pos_vals, gold_attn_vals, correct_vals, ["beginning", "middle", "end"]):
    marker = "o" if c else "X"
    ax.scatter(x, y, s=200, marker=marker,
               color=SCENARIO_COLORS[s],
               edgecolors="black", linewidths=1.2,
               zorder=5, label=f"{s} (pos {x:.1f})")
ax.plot(gold_pos_vals, gold_attn_vals, "--", color="#999", linewidth=1, zorder=4)
ax.set_xlabel("Position of gold evidence in context (out of 20)", fontsize=12)
ax.set_ylabel("Total attention to gold evidence", fontsize=12)
ax.set_title("Attention to gold evidence vs. position\n(○=correct, ✗=wrong)", fontsize=11)
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(fontsize=9)

# Panel B: full chunk-level attention curve for each scenario (smoothed)
ax = axes[1]
chunk_x = np.arange(1, N_TOTAL_CHUNKS + 1)
for scenario_name, res in results.items():
    ax.plot(
        chunk_x, res["chunk_attn"],
        color=SCENARIO_COLORS[scenario_name],
        marker="o", markersize=5, linewidth=1.5, alpha=0.85,
        label=f"{scenario_name} (gold @ {res['gold_positions'][0]+1}–{res['gold_positions'][1]+1})"
               + (" ✓" if res["numeric_match"] else " ✗"),
    )
    # Mark gold positions
    for gi in res["gold_positions"]:
        ax.axvline(gi + 1, color=SCENARIO_COLORS[scenario_name],
                   linestyle=":", linewidth=1.2, alpha=0.6)

ax.set_xlabel("Chunk position in context", fontsize=12)
ax.set_ylabel("Summed attention to chunk", fontsize=12)
ax.set_title("Positional attention profile across three scenarios", fontsize=11)
ax.set_xticks(chunk_x)
ax.set_xticklabels([str(i) for i in chunk_x], fontsize=7)
ax.legend(fontsize=9)

plt.suptitle(
    f"Lost-in-the-Middle Positional Bias — Qwen2.5-14B-Instruct\n"
    f"FinanceBench {SAMPLE_ID} | Reference answer: {reference_answer}",
    fontsize=12, y=1.01,
)
plt.tight_layout()
out_path = OUT_DIR / "ucurve.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ---- 6d. Summary text -------------------------------------------------
summary_lines = [
    f"FinanceBench ID : {SAMPLE_ID}",
    f"Document        : ADOBE_2015_10K",
    f"Question        : {question[:100]}...",
    f"Reference answer: {reference_answer}",
    "",
]
for sn, res in results.items():
    gp = res["gold_positions"]
    gold_attn = float(res["chunk_attn"][list(gp)].sum())
    summary_lines.append(
        f"  [{sn:9s}] gold pos {gp[0]+1:2d}–{gp[1]+1:2d}/20  |  "
        f"gold attn={gold_attn:.5f}  |  "
        f"correct={'YES' if res['numeric_match'] else 'NO '} |  "
        f"answer: {res['generated_answer'][:80]}..."
    )

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUT_DIR}")
