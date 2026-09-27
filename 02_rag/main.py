"""
RAG baseline — IEEE-CIS Technical Challenge on Domain-Specific Question
Answering with LLMs.

Blind retrieval over the whole corpus: every PDF in corpus_dir is parsed with
Docling, chunked and embedded once per run, then for each question every
chunk is ranked by cosine similarity and the best-ranked ones are packed
into a fixed context budget before asking the base model. No document_id is
known to this code — the same "search has to find it" design as a real
participant would face.

Needs a local copy of Docling's layout model and of the embedding model next
to this file (see build.sh): the evaluation environment has no internet
access, so neither can be downloaded there the first time they're used.
"""

import re
from pathlib import Path

import numpy as np
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)

MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# A local copy, not the "sentence-transformers/all-MiniLM-L6-v2" repo id:
# same reason as the Docling model below — small, but still a model the
# evaluation environment doesn't have pre-downloaded, and there's no
# internet there to fetch it by name (see build.sh).
EMBED_MODEL_PATH = str(Path(__file__).parent / "embed_model")

# Fixed context budget, in tokens. Deliberately NOT "as much as fits the
# model's real context window" (Olmo-3.1-32B-Instruct's is 65536): measured
# directly, filling anywhere near that runs the GPU (80 GB) out of memory
# during prefill, and even a context that does fit paired with a long real
# answer can take minutes for a single question. 10,000 keeps every
# question fast and safe — tested end to end against real questions.
MAX_CONTEXT_TOKENS = 10_000

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}


def _docling_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(Path(__file__).parent / "docling_models"),
        do_ocr=False,              # the corpus is born-digital text, not scans
        do_table_structure=False,  # avoids needing cv2/libGL, not in the image
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})


def _chunk_markdown(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Greedily pack paragraphs into ~chunk_size-char chunks, so cuts land on
    paragraph boundaries instead of mid-sentence, carrying the tail of each
    chunk into the next one for a bit of overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _build_index(corpus_dir: str):
    """Parse + chunk + embed every PDF in corpus_dir. Runs once per submission."""
    converter = _docling_converter()
    embedder = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")

    all_chunks = []
    pdfs = sorted(Path(corpus_dir).glob("*.pdf"))
    for pdf_path in pdfs:
        try:
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
        except Exception as e:
            # One bad PDF shouldn't take down the whole run.
            print(f"[rag] skipping {pdf_path.name}: {e}")
            continue
        all_chunks.extend(_chunk_markdown(text))

    print(f"[rag] parsed {len(pdfs)} PDFs into {len(all_chunks)} chunks")
    embeddings = embedder.encode(
        all_chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=64,
    )
    return all_chunks, np.asarray(embeddings, dtype=np.float32), embedder


def _load(model_id: str, corpus_dir: str):
    if model_id not in _cache:
        chunks, embeddings, embedder = _build_index(corpus_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        _cache[model_id] = (tokenizer, model, chunks, embeddings, embedder)
    return _cache[model_id]


def _build_context(question, chunks, embeddings, embedder, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
    """As many of the top-ranked chunks (blind, corpus-wide) as fit in
    max_tokens. Falls back to a truncated slice of the single best chunk
    when even that alone doesn't fit."""
    query_emb = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    order = np.argsort(-(embeddings @ query_emb))

    selected, used = [], 0
    for i in order:
        chunk = chunks[i]
        n = len(tokenizer.encode(chunk))
        if used + n > max_tokens:
            if not selected and n > 0:
                ratio = max_tokens / n
                selected.append(chunk[: int(len(chunk) * ratio)])
            break
        selected.append(chunk)
        used += n

    if not selected:
        return ""
    body = "\n\n".join(f"[Fragment {i}]\n{c}" for i, c in enumerate(selected, 1))
    return f"Context extracted from a blind search over the corpus:\n\n{body}\n\n---\n\n"


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    tokenizer, model, chunks, embeddings, embedder = _load(model_id, corpus_dir)
    context = _build_context(question, chunks, embeddings, embedder, tokenizer)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context + question},
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
