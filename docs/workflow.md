# Development and experiments

Keep `01_zero_shot/` and `02_rag/` unchanged. Use `feat/<feature-name>` branches
for new work; keep reusable implementation in Python modules, not notebooks.
No local ML dependencies or weights are needed for the checks below:

```sh
uv venv --python 3.12
uv run --no-project python -m unittest discover -s tests -v
uv run --no-project python -m experiments.package zero_shot 01_zero_shot_submission.zip
```

## Verified submission interface

Source inspected: fork commit `5e596be2f9d9d5ff8d8c89bf8bd8d43b42ec8e8a`,
README, both `main.py` and `build.sh` files, participant pins, all 25 sample
QA pairs, sites list, ignore rules and license. No ingestion/scoring harness or
separate submission-contract document is included in this repository.

The baselines define:

```python
def answer(question: str, corpus_dir: str, model_id: str) -> str:
    ...
```

- `question` is the question text. `corpus_dir` is a filesystem path;
  zero-shot ignores it; RAG scans only its immediate `*.pdf` children.
  `model_id` is passed unchanged to Hugging Face tokenizer/model loaders.
- Return a Python string containing the decoded generated answer, with special
  tokens removed and surrounding whitespace stripped. There is no JSON envelope
  or output-file protocol in the baseline code. The local runner additionally
  rejects empty answers as a quality check.
- Comments specify one call per question in a persistent process. Both modules
  cache by **model_id only**. Restart between experiments, particularly when
  changing corpus or retrieval settings; the runner rejects a populated cache.
- Loading uses `dtype=torch.bfloat16`, `device_map="cuda"`, `eval()` and
  `torch.no_grad()`. No quantization or CPU fallback is implemented. Both use
  chat templates, greedy generation (`do_sample=False`), and at most 2048 new
  tokens. The notebook deliberately uses 128 for inexpensive smoke tests.
- RAG parses PDFs with Docling (OCR and table structure disabled), skips failed
  PDFs, packs paragraphs into approximately 800-character chunks with a
  150-character tail overlap (long paragraphs can exceed that size), embeds on
  CPU using local MiniLM, and ranks normalized vectors by dot product. It has
  no fixed top-k: it adds ranked chunks until the 10,000-token budget would be
  exceeded, then stops. The first oversized chunk is approximated by character
  truncation. Fragment labels and prompt text are outside that token budget.
  No question-to-document mapping is available. An empty index is not handled
  reliably by the baseline; provide parseable PDFs for RAG tests.
- RAG loads its local `embed_model/` and `docling_models/` beside `main.py`.
  Its build script replaces those directories and downloads their weights.
  The large generation model is not included in either ZIP.
- Both build scripts put `main.py` at the ZIP root. RAG also includes the two
  auxiliary model directories. `experiments.package` reproduces that layout
  without downloading anything; it checks asset presence, not model integrity.
- `requirements-participant.txt` is the compatibility authority, not Colab's
  installed packages. It pins torch 2.8.0, transformers 5.17.0, accelerate
  1.15.0, Docling 2.127.0 and sentence-transformers 6.0.1, among others.
  The README says no other dependencies can be added at evaluation time.
  vLLM and flash-attn are explicitly absent; use transformers/accelerate.

The [organizer's submission page](https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/)
checked on 2026-10-22 specifies a single H100, a one-hour total wall-clock
budget including loading/indexing/inference, no internet, a 1 GB submission
limit and one submission per day per team. It names OLmo 3.1 32B Instruct as
the fixed evaluation model. The baseline comments describe an 80 GB GPU and
a 65,536-token model window; neither is a safe usable-prompt memory budget.

The exact server-supplied model identifier/cache path, harness import details,
question order/count per run, scoring implementation, RAM/CPU/disk allocation,
and interpretation of the 1 GB limit are not established by this code. The
[CodaBench page](https://www.codabench.org/competitions/17954/) could not be
retrieved during inspection. Verify platform-specific details there before
uploading; do not infer them from sample JSON or local tests.

## Colab sequence

1. Commit and push the feature branch when ready to make it accessible to Colab.
   Open `notebooks/baseline_colab.ipynb` in Colab and select a GPU runtime.
2. Set `REF` to the pushed commit SHA for reproducibility. The first cell checks
   GPU name, total/free memory and checks out that source. No weights are loaded.
3. Leave all opt-ins false initially. The notebook creates a uv environment and
   installs the selected baseline's direct dependencies at participant-pinned
   versions, then checks dependency consistency and runs CPU tests. It does not
   install the entire participant environment. Unavailable pins are a blocking
   compatibility issue: do not silently loosen them. Transitive versions are
   captured by `pip freeze`; this is not a full reproduction of the server image.
4. Keep the default small substitute model for a smoke test. It uses the same
   baseline inference path but cannot establish official answer quality.
   Check BF16 support and available memory. The unchanged baseline requires
   BF16; older GPU allocations may require a different Colab GPU.
5. For RAG, supply your own PDFs under `/content/corpus`. The private corpus is
   unavailable; `sample/sites.txt` is a source-domain list, not a document set.
   Set `STRATEGY='rag'`, then explicitly enable `ALLOW_MODEL_DOWNLOADS` and
   `PREPARE_RAG_ASSETS` to run the original RAG build script **in Colab**.
6. Explicitly set `ALLOW_MODEL_DOWNLOADS=True` and `RUN_INFERENCE=True` only when
   ready. The default runs two sample questions. Never choose the 32B model on
   the laptop. BF16 weights alone are about 64 GB (decimal), before activations,
   KV cache and overhead; smaller Colab allocations need the substitute model.
   Use the exact official identifier only once confirmed and sufficient memory
   has been provisioned. No automatic large-model fallback or retry is attempted.
7. Download `results/run.json` before the runtime expires. It follows
   `experiments/results.schema.json`: model, source/environment, prompt,
   retrieval configuration, corpus file hashes, answers/references and timing.
   First-call latency includes lazy model loading and RAG indexing; later calls
   are warm. CUDA allocated/reserved peaks are PyTorch allocator measurements,
   not total device usage or CPU RAM. References never enter inference prompts.
8. Restart the runtime for each changed configuration. Compare identical sample
   subsets and corpora. These 25 illustrative samples are not the hidden benchmark.

## Preparing a submission (no automatic upload)

In the Colab uv environment, after CPU tests and a real smoke test:

```sh
.venv/bin/python -m experiments.package zero_shot 01_zero_shot_submission.zip
# RAG only after explicitly preparing auxiliary models in Colab:
.venv/bin/python -m experiments.package rag 02_rag_submission.zip
unzip -l 02_rag_submission.zip
```

Alternatively use the unchanged organizer scripts: `bash 01_zero_shot/build.sh`
or `bash 02_rag/build.sh` with `.venv/bin` on PATH. The RAG script downloads assets
every time, so use the Python packager to reuse prepared assets. Do not package
notebooks, experiment logs, the Python environment or generation weights.
Inspect archive size against the published limit and verify the root layout.
Test prepared RAG assets with Hugging Face offline flags before upload; CPU mocks
cannot prove auxiliary model completeness or actual offline initialization.
Recheck participant pins and the platform contract. Manually upload the intended
archive only when ready; nothing in this workflow submits to CodaBench.

## Validation limits

CPU tests execute the real baseline `answer` functions extracted with Python's
AST against fake tokenizer/model boundaries; they check signatures, prompt
composition, generated-token slicing and stripping. They also check ZIP layout,
missing RAG assets, and notebook syntax/defaults. They do not import the full ML
stack, download weights, measure GPU feasibility or reproduce official scoring.
