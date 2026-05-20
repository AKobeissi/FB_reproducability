import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/u/kobeissa/Documents/thesis/experiments/FB_reproducability")
DATA_FILE = PROJECT_ROOT / "data/financebench_open_source.jsonl"

def load_fb_data():
    fb_map = {}
    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            # Use question as key for mapping
            fb_map[r["question"]] = r
    return fb_map

def convert_oracle(src_path, dest_name, fb_map):
    with open(src_path) as f:
        data = json.load(f)
    
    converted = []
    for res in data["results"]:
        q = res["question"]
        if q not in fb_map:
            print(f"Warning: question not found in FB map: {q[:50]}...")
            continue
        
        fb_item = fb_map[q]
        
        item = {
            "financebench_id": fb_item.get("financebench_id"),
            "question": q,
            "reference_answer": fb_item.get("answer"),
            "question_type": fb_item.get("question_type"),
            "doc_name": fb_item.get("doc_name"),
            "doc_link": fb_item.get("doc_link"),
            "gold_evidence_segments": fb_item.get("evidence"), # Format might differ slightly but gen_comparison might not care
            "retrieved_chunks": res.get("retrieved_chunks", [])
        }
        converted.append(item)
    
    dest_path = PROJECT_ROOT / f"baselines/results/predictions/{dest_name}"
    os.makedirs(dest_path.parent, exist_ok=True)
    with open(dest_path, "w") as f:
        json.dump(converted, f, indent=2)
    print(f"Converted {len(converted)} samples to {dest_path}")

fb_map = load_fb_data()

# Oracle Doc
convert_oracle(
    "outputs/oracle_doc/oracle_doc/20260210/oracle_doc_20260210_162834.json",
    "oracle_doc_retrieval.json",
    fb_map
)

# Oracle Page
convert_oracle(
    "outputs/oracle_page/oracle_page/20260210/oracle_page_20260210_174045.json",
    "oracle_page_retrieval.json",
    fb_map
)
