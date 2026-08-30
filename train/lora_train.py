"""ML-Loop: autonomous LoRA fine-tuning on GitHub Actions CPU."""
import logging
import os

import requests
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

try:
    from .data import fmt, load_texts
except ImportError:
    from data import fmt, load_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml-loop")

OUT_DIR = "adapter"
MODEL = "Qwen/Qwen2.5-0.5B"
MIN_EXAMPLES = 20


def perplexity(model, tokenizer, ds_eval, max_samples: int = 20) -> float:
    """Mean loss over up to max_samples eval examples."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ex in list(ds_eval)[:max_samples]:
            enc = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=256)
            out = model(**enc, labels=enc["input_ids"])
            total += out.loss.item()
            count += 1
    return total / max(count, 1)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def publish_to_hf(folder: str, repo_id: str, token: str) -> None:
    HfApi().upload_folder(folder_path=folder, path_in_repo=".", repo_id=repo_id, token=token)


def notify_telegram(text: str) -> None:
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not (token and chat):
        logger.info("TG_TOKEN/TG_CHAT not set, skipping notification")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.error("Telegram notification failed: %s", exc)


def main() -> None:
    texts = load_texts()
    logger.info("Collected examples: %d", len(texts))
    if len(texts) < MIN_EXAMPLES:
        logger.warning("Not enough data (%d < %d), skipping training", len(texts), MIN_EXAMPLES)
        raise SystemExit(0)

    ds = Dataset.from_list([fmt(t) for t in texts])
    split = ds.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

    loss_before = perplexity(model, tokenizer, split["test"])
    logger.info("Eval loss BEFORE: %.4f", loss_before)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=256,
        dataset_text_field="text",
    )
    trainer.train()

    loss_after = perplexity(trainer.model, tokenizer, split["test"])
    logger.info("Eval loss: %.4f -> %.4f", loss_before, loss_after)

    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    try:
        publish_to_hf(OUT_DIR, os.environ["HF_REPO"], os.environ["HF_TOKEN"])
        logger.info("Adapter published: %s", os.environ["HF_REPO"])
    except Exception as exc:
        logger.error("Publish failed: %s", exc)
        notify_telegram(f"ML-Loop: publish failed: {exc}")
        raise

    notify_telegram(
        f"ML-Loop завершён\nПримеров: {len(texts)}\n"
        f"Eval loss: {loss_before:.4f} -> {loss_after:.4f}\n"
        f"Адаптер: {os.environ.get('HF_REPO')}"
    )


if __name__ == "__main__":
    main()