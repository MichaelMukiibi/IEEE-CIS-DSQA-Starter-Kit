"""
Zero-shot baseline — IEEE-CIS Technical Challenge on Domain-Specific
Question Answering with LLMs.

No retrieval: answers every question using only the base model's own knowledge,
exactly as it comes out of the box. This is the reference floor every other submission 
should beat.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)

MAX_NEW_TOKENS = 2048

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}

def _load(model_id: str):
    if model_id not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        _cache[model_id] = (tokenizer, model)
    return _cache[model_id]

def answer(question: str, corpus_dir: str, model_id: str) -> str:
    tokenizer, model = _load(model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()