"""Data loading and formatting for ML-Loop (stdlib only, unit-testable)."""
import glob
import json
import logging

logger = logging.getLogger("ml-loop")


def load_texts(data_dir: str = "data") -> list:
    """Load 'text' fields from *.jsonl, skipping malformed lines."""
    texts = []
    for path in sorted(glob.glob(f"{data_dir}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping bad line in %s: %s", path, exc)
    return texts


def fmt(text: str) -> dict:
    """Wrap raw text into an instruction-style training sample."""
    return {"text": f"Проанализируй текст и выдели главное:\n{text}\nГлавное:"}