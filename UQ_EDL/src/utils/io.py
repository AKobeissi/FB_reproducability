import json
import yaml
import pathlib
from typing import Any, Iterator


def load_jsonl(path: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_json(path: str | pathlib.Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: str | pathlib.Path, indent: int = 2) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_yaml(path: str | pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def iter_jsonl(path: str | pathlib.Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
