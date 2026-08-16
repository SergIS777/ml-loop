"""ML-Loop: LoRA fine-tuning Qwen2.5-0.5B на собранных данных.
Запускается в GitHub Actions (CPU, 7GB RAM)."""
import json, os, glob
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

DATA_DIR = "data"
OUT_DIR = "adapter"
MODEL = "Qwen/Qwen2.5-0.5B"

files = sorted(glob.glob(f"{DATA_DIR}/*.jsonl"))
if not files:
    print("[ml-loop] нет данных, пропускаем")
    raise SystemExit(0)

texts = []
for f in files:
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])

print(f"[ml-loop] собрано примеров: {len(texts)}")
if len(texts) < 20:
    print(f"[ml-loop] мало данных ({len(texts)} < 20), пропускаем обучение")
    raise SystemExit(0)

def fmt(t):
    return {"text": f"Проанализируй текст и выдели главное:\n{t}\nГлавное:"}

ds = Dataset.from_list([fmt(t) for t in texts])
split = ds.train_test_split(test_size=0.1, seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

def perplexity(m, ds_eval):
    m.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ex in list(ds_eval)[:20]:
            enc = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=256)
            out = m(**enc, labels=enc["input_ids"])
            total += out.loss.item()
            count += 1
    return total / max(count, 1)

loss_before = perplexity(model, split["test"])
print(f"[ml-loop] eval loss ДО: {loss_before:.4f}")

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

loss_after = perplexity(trainer.model, split["test"])
print(f"[ml-loop] eval loss: {loss_before:.4f} -> {loss_after:.4f}")

trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path=OUT_DIR, path_in_repo=".",
    repo_id=os.environ["HF_REPO"], token=os.environ["HF_TOKEN"],
)
print(f"[ml-loop] адаптер опубликован: {os.environ['HF_REPO']}")

import requests
tg_token, tg_chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
if tg_token and tg_chat:
    msg = (f"🤖 ML-Loop завершён\nПримеров: {len(texts)}\n"
           f"Eval loss: {loss_before:.4f} → {loss_after:.4f}\n"
           f"Адаптер: {os.environ['HF_REPO']}")
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                  json={"chat_id": tg_chat, "text": msg})
