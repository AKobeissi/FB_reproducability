import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/u/kobeissa/Documents/thesis/experiments/FB_reproducability")

EXPERIMENTS = [
    ("Dense BGE-M3", "dense_bge_m3"),
    ("BGE + MultiHyDE + RR", "multi_hyde_rr"),
    ("BGE + MultiHyDE + FTRR", "multi_hyde_ftrr"),
    ("Oracle Doc", "oracle_doc"),
    ("Oracle Page", "oracle_page")
]
MODELS = ["Qwen2.5-7B", "Llama3.1-8B"]

def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def format_table(headers, rows):
    if not rows:
        return "No data available."
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    fmt = " | ".join(["{:<" + str(w) + "}" for w in widths])
    sep = "-|-".join(["-" * w for w in widths])
    
    output = "| " + fmt.format(*headers) + " |\n"
    output += "| " + sep + " |\n"
    for row in rows:
        output += "| " + fmt.format(*[str(x) if x is not None else "-" for x in row]) + " |\n"
    return output

def aggregate_all():
    # 1. OVERALL TABLE
    overall_headers = ["Retrieval Method", "Model", "ROUGE-L", "BERTScore", "NumMatch(metrics)", "NumMatch(all)"]
    overall_rows = []
    
    # 2. BY QUESTION TYPE
    qtype_headers = ["Retrieval Method", "Question Type", "Metric", "Qwen2.5-7B", "Llama3.1-8B"]
    qtype_rows = []
    
    # 3. BY DOC TYPE
    dtype_headers = ["Retrieval Method", "Doc Type", "Metric", "Qwen2.5-7B", "Llama3.1-8B"]
    dtype_rows = []

    for label, exp_id in EXPERIMENTS:
        base_path = PROJECT_ROOT / f"outputs/gen_comparison_{exp_id}"
        
        summary = load_json(base_path / "summary.json")
        if summary:
            for model in MODELS:
                if model in summary:
                    m = summary[model]
                    overall_rows.append([
                        label, model, 
                        f"{m.get('rougeL', 0):.4f}", 
                        f"{m.get('bertscore_f1', 0):.4f}", 
                        f"{m.get('numeric_match', 0):.4f}", 
                        f"{m.get('numeric_match_all', 0):.4f}"
                    ])

        # Question Type
        q_data = load_json(base_path / "by_question_type.json")
        if q_data:
            for qtype in ["metrics-generated", "domain-relevant", "novel-generated"]:
                q_m0 = q_data.get(MODELS[0], {}).get(qtype, {})
                q_m1 = q_data.get(MODELS[1], {}).get(qtype, {})
                
                qtype_rows.append([label, qtype, "ROUGE-L", f"{q_m0.get('rougeL', 0):.4f}", f"{q_m1.get('rougeL', 0):.4f}"])
                qtype_rows.append([label, qtype, "NumMatch", f"{q_m0.get('numeric_match_all', 0):.4f}", f"{q_m1.get('numeric_match_all', 0):.4f}"])

        # Doc Type
        d_data = load_json(base_path / "by_doc_type.json")
        if d_data:
            for dtype in ["10k", "10q", "8k", "Earnings"]:
                d_m0 = d_data.get(MODELS[0], {}).get(dtype, {})
                d_m1 = d_data.get(MODELS[1], {}).get(dtype, {})
                
                dtype_rows.append([label, dtype, "ROUGE-L", f"{d_m0.get('rougeL', 0):.4f}", f"{d_m1.get('rougeL', 0):.4f}"])
                dtype_rows.append([label, dtype, "NumMatch", f"{d_m0.get('numeric_match_all', 0):.4f}", f"{d_m1.get('numeric_match_all', 0):.4f}"])

    print("\n### 1. Overall Generative Performance")
    print(format_table(overall_headers, overall_rows))
    
    print("\n### 2. Breakdown by Question Type")
    print(format_table(qtype_headers, qtype_rows))
    
    print("\n### 3. Breakdown by Document Type")
    print(format_table(dtype_headers, dtype_rows))

if __name__ == "__main__":
    aggregate_all()
