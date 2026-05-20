"""PyTorch Dataset for cross-encoder training pairs."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.io import load_jsonl
from src.utils.text import build_model_input, truncate_text


class PairDataset(Dataset):
    def __init__(
        self,
        pairs_path: str | pathlib.Path,
        tokenizer_name: str,
        max_length: int = 512,
        use_metadata: bool = True,
    ):
        self.pairs = load_jsonl(pairs_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.use_metadata = use_metadata

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        question = pair["question"]
        candidate_text = pair.get("candidate_text", "")
        meta = pair.get("metadata", {}) if self.use_metadata else None

        input_text = build_model_input(question, truncate_text(candidate_text, 3000), meta)
        # Tokenize as single sequence (query + context already merged)
        enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "label": torch.tensor(pair["label"], dtype=torch.long),
            "qid": pair.get("qid", ""),
            "candidate_id": pair.get("candidate_id", ""),
        }
