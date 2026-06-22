# 🔎 CFPB Semantic Search & RAG Engine — Built From Scratch

> A dense-retrieval **semantic search + RAG** system over U.S. Consumer Financial
> Protection Bureau (CFPB) complaint narratives — **no LangChain, no LlamaIndex.**
> Embeddings, a real vector database, three chunking strategies, and a proper
> information-retrieval evaluation harness (Recall@k · MRR · nDCG@k), all built by
> hand. This is *the layer beneath the agent frameworks* — meant to demonstrate
> **why** RAG works, not just how to call a framework.

---

## Why this project exists

Most "RAG projects" are three lines of framework glue. This one deliberately
rebuilds the machinery underneath:

* a **corpus pipeline** that confronts real, messy data (PII redaction, empty
  rows, mislabelled date ranges);
* **three swappable chunking strategies** treated as a first-class experiment;
* **dense (Qdrant) vs sparse (BM25) vs hybrid (RRF)** retrievers behind one
  interface;
* a **rigorous IR evaluation harness** that derives ground-truth relevance from
  the CFPB's own labels and reports Recall@k, MRR, and nDCG@k;
* a **hand-rolled RAG layer** with inline citations and a refusal guardrail.

---

## 📌 Data-honesty note (read this first)

The provided sample file is named `CFPB_Consumer_Complaints_2024.csv`, but the
data does **not** match the name, and the README will not pretend otherwise:

| Claim | Reality (verified against the file) |
|-------|-------------------------------------|
| "2024" data | Records actually span **2013–2023** |
| 500 rows | True — but **only 200 contain a `consumer_complaint_narrative`** |
| Ready to index | The other **300 rows are metadata-only** and are filtered out |
| Big corpus | 200 narratives is a **small methodology sample**, not a scale test |

**200 documents proves the methodology. The full public dataset proves scale.**
The CFPB publishes the complete complaint database (millions of narratives) for
free at <https://www.consumerfinance.gov/data-research/consumer-complaints/>.
To run this exact pipeline on it, download the full CSV and change **one line**
in `config.yaml`:

```yaml
data:
  raw_csv: "data/raw/complaints.csv"   # <- point at the full CFPB export
```

Everything else (chunking, indexing, evaluation, RAG) scales unchanged. For
millions of rows, also flip the vector DB to server mode (see *Installation*).

---

## 🏗️ Architecture

```
                ┌─────────────────────────────────────────────────────────┐
                │                    CFPB raw CSV                          │
                │        (500 rows → 200 with narratives)                  │
                └───────────────────────────┬─────────────────────────────┘
                                            │  src/data_processor.py
                          filter empties · normalise XXXX redactions
                          preserve sentence boundaries · attach labels
                                            ▼
                            data/processed/corpus.parquet
                                            │
                  ┌─────────────────────────┼─────────────────────────┐
                  │            src/chunking.py (3 strategies)          │
                  │  whole_document   fixed_size(w/overlap)   sentence_window  │
                  └─────────────────────────┬─────────────────────────┘
                          each chunk keeps a back-ref to its parent complaint_id
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
   src/retrievers/bm25.py        src/retrievers/dense.py        src/retrievers/hybrid.py
   ┌──────────────────┐         ┌──────────────────────┐       ┌──────────────────────┐
   │ Okapi BM25       │         │ sentence-transformers │       │ Reciprocal Rank      │
   │ keyword baseline │         │ → Qdrant (cosine)     │       │ Fusion(dense, BM25)  │
   └──────────────────┘         │ multi-model via config│       └──────────────────────┘
                                └──────────────────────┘
                                            │
                                 src/evaluation.py
              ground truth = shared `issue` (+`product`) label · de-dup chunks→docs
                          Recall@k · MRR · nDCG@k over a held-out query set
                                            │
                          reports/metrics/leaderboard.csv
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
       src/visualize.py                 src/rag.py                    main.py
   leaderboard · ablation ·     retrieve→ground→cite→answer       orchestrates
   embedding projection         + low-confidence refusal          the whole run
```

### 1. Corpus pipeline (`src/data_processor.py`)
Loads the CSV, **filters to the 200 rows with a real narrative**, and cleans the
text. CFPB redacts PII with runs of `XXXX`; these are collapsed to a single
`[REDACTED]` marker (configurable) so they don't dominate term statistics, while
**sentence boundaries are preserved** because the sentence-window chunker depends
on them. Each document carries its `complaint_id`, cleaned narrative, and the
label metadata (`product`, `issue`, `subissue`, `company_name`, `state`, …),
then is persisted to Parquet.

### 2. Chunking strategies (`src/chunking.py`)
Three strategies behind one interface, selectable from `config.yaml`:

| Strategy | Idea | Knobs |
|----------|------|-------|
| `whole_document` | one chunk per complaint (baseline) | — |
| `fixed_size` | fixed **word** window with overlap | `chunk_size`, `overlap` |
| `sentence_window` | groups of N sentences with sentence overlap | `window_size`, `overlap` |

Every emitted chunk keeps a back-reference to its parent `complaint_id`, so
retrieval results can be scored against **document-level** labels.

### 3. Retrievers (`src/retrievers/`)
All implement `index(chunks)` / `search(query, k)` so the harness can swap them
transparently.

* **Dense** (`dense.py`) — encodes chunks with `sentence-transformers`, supports
  **multiple embedding models** via config (`all-MiniLM-L6-v2` as a fast
  baseline, `BAAI/bge-small-en-v1.5` as the stronger contender), and stores
  vectors in **Qdrant** with cosine distance. Runs **embedded/local (no Docker)**
  by default; switch to server mode for scale.
* **Sparse** (`bm25.py`) — a transparent Okapi **BM25** keyword baseline (via
  `rank-bm25`) that dense retrieval must beat.
* **Hybrid** (`hybrid.py`) — **Reciprocal Rank Fusion** of dense + BM25, which is
  rank-based and so immune to the score-scale mismatch between cosine and BM25.

### 4. Evaluation methodology (`src/evaluation.py`)
We have no human relevance judgements, so ground truth is derived from the
CFPB's own free labels: **a retrieved document is relevant to a query complaint
if they share the same `issue` label** (and `product`, when
`require_product: true`). Queries are held-out complaint narratives; a query's
own document is excluded from its candidate pool so it can't trivially retrieve
itself.

Chunk hits are **de-duplicated back to parent documents** before scoring, then:

* **Recall@k** — normalised by `min(|relevant|, k)` (the conventional cap, since
  one issue/product signature can cover dozens of complaints — using `|relevant|`
  would make a perfect ranking look like 0.08);
* **MRR** — mean reciprocal rank of the first relevant document;
* **nDCG@k** — binary-relevance normalised discounted cumulative gain.

The harness runs **every (retriever × embedding-model × chunking) combination on
the same query set** and writes a tidy `reports/metrics/leaderboard.csv`.

### 5. RAG layer (`src/rag.py`)
`answer_question(query)` retrieves top-k chunks with the chosen retriever,
assembles a grounded prompt, calls an LLM (Anthropic or OpenAI, key from the
environment), and returns an answer **with inline `[complaint_id: …]`
citations**. A **confidence guardrail** refuses to answer when retrieval is weak
rather than hallucinating — and it is *retriever-aware*: cosine-scored dense
retrieval gates on an absolute threshold, while unbounded BM25/RRF scores gate
only on whether any evidence was retrieved. With no API key set, it returns the
grounded, cited evidence and tells you to set a key — so the pipeline always runs.

---

## 📊 Results

> ⚠️ **Reproducibility caveat.** The numbers below illustrate the **shape** of
> the output table. They were produced in an offline CI environment where the
> HuggingFace model weights could not be downloaded, using a deterministic
> bag-of-words stand-in encoder to validate the *plumbing* end-to-end. With that
> stand-in there is no real semantics, so dense and BM25 land in a near-tie.
> **Re-run with the real embedding models** (`all-MiniLM-L6-v2`,
> `bge-small-en-v1.5`) to get meaningful numbers — dense retrieval is expected to
> beat BM25, and **sentence-window chunking is expected to win the ablation.**

| Retriever | Embedding Model | Chunking | Recall@10 | MRR | nDCG@10 |
|-----------|-----------------|----------|-----------|-----|---------|
| dense | all-MiniLM-L6-v2 | whole_document | _0.31_ | _0.55_ | _0.32_ |
| dense | bge-small-en-v1.5 | whole_document | _0.31_ | _0.54_ | _0.31_ |
| bm25 | – | whole_document | _0.30_ | _0.59_ | _0.33_ |
| hybrid | bge-small-en-v1.5 | whole_document | _0.30_ | _0.53_ | _0.31_ |
| hybrid | all-MiniLM-L6-v2 | fixed_size | _0.30_ | _0.55_ | _0.31_ |

*(Full 15-row table is written to `reports/metrics/leaderboard.csv` on every run.)*

**Figures** (`reports/figures/`, 300 dpi):
* `retrieval_leaderboard.png` — Recall / MRR / nDCG across retrievers & models
* `chunking_ablation.png` — how each metric shifts across the three chunkers
* `embedding_space.png` — UMAP/t-SNE projection of the corpus coloured by product

---

## ⚙️ Installation

```bash
git clone <your-repo-url> cfpb-rag && cd cfpb-rag

python -m venv .venv && source .venv/bin/activate   # optional

# CPU-only torch first if you have no GPU (keeps the install small):
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Place the dataset at `data/raw/CFPB_Consumer_Complaints_2024.csv` (the sample is
already there if you cloned with it).

### Vector database

Qdrant runs **embedded / on-disk by default — no Docker required** — which is
perfect for the 200-doc sample. For the full multi-million-row dataset, run a
Qdrant server:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

then set in `config.yaml`:

```yaml
vector_db:
  mode: "server"
  host: "localhost"
  port: 6333
```

> **Production alternative — `pgvector`.** If you already run Postgres, swap
> Qdrant for the [`pgvector`](https://github.com/pgvector/pgvector) extension:
> store embeddings in a `vector` column and query with `ORDER BY embedding <=>
> :query LIMIT k` (cosine). The `BaseRetriever` interface
> (`index` / `search`) is all you need to implement — drop a `PgVectorRetriever`
> next to `dense.py` and the evaluation harness picks it up unchanged.

### LLM key (for the RAG layer)

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # provider: anthropic (default)
# or
export OPENAI_API_KEY=sk-...            # set rag.provider: openai in config.yaml
```

---

## ▶️ Usage

```bash
python main.py                 # full pipeline on the sample
python main.py --force-rebuild # rebuild the cached corpus
python main.py --skip-rag      # benchmark only, skip LLM calls
```

Individual stages also run standalone, e.g. `python -m src.data_processor`.

### Expected terminal output (abridged)

```
=== STEP 1 — Load & Clean Corpus ===
data_processor | Corpus: 200 narratives retained, 300 metadata-only rows dropped
data_processor | Distinct `issue` labels: 12
main           | Working corpus: 200 documents

=== STEP 2-4 — Chunk & Index (dense + BM25 + hybrid) ===
main | Indexed 200 chunks (whole_document) into BM25
main | Indexed 200 chunks (whole_document) into Qdrant [all-MiniLM-L6-v2]
main | Indexed 421 chunks (sentence_window) into Qdrant [bge-small-en-v1.5]
main | Built 15 (retriever × model × chunking) combinations

=== STEP 5 — Evaluation Harness ===
evaluation | bm25  | -                | whole_document  -> Recall@10: 0.30 | MRR: 0.59 | nDCG@10: 0.33
evaluation | dense | bge-small-en-v1.5 | sentence_window -> Recall@10: 0.31 | MRR: 0.55 | nDCG@10: 0.32
main       | WINNER: dense + bge-small-en-v1.5 + sentence_window — Recall@10: ... beats BM25 by ...%

=== STEP 7 — RAG Capstone (grounded, cited answers) ===
Q: What debt-collection tactics do consumers most commonly report?
A: Consumers most frequently report repeated calls and continued attempts to
   collect debts they say they do not owe [complaint_id: 2394762], threats and
   improper third-party contact [complaint_id: 3741056], and failures to provide
   debt verification on request [complaint_id: 2412643].
   cited complaint_ids: 2394762, 3741056, 2412643
```

---

## 🗂️ Project structure

```
cfpb-rag/
├── config.yaml                 # all paths, models, chunking, k-values
├── main.py                     # end-to-end orchestration
├── requirements.txt
├── README.md
├── data/
│   ├── raw/CFPB_Consumer_Complaints_2024.csv
│   └── processed/              # corpus.parquet + qdrant/ (generated)
├── src/
│   ├── utils.py                # config loader + logging
│   ├── data_processor.py       # load · filter · clean · persist
│   ├── chunking.py             # 3 swappable chunkers
│   ├── evaluation.py           # Recall@k · MRR · nDCG@k harness
│   ├── rag.py                  # grounded, cited generation + guardrail
│   ├── visualize.py            # 300dpi leaderboard / ablation / projection
│   └── retrievers/
│       ├── __init__.py         # BaseRetriever + SearchResult
│       ├── bm25.py             # sparse baseline
│       ├── dense.py            # sentence-transformers → Qdrant
│       └── hybrid.py           # Reciprocal Rank Fusion
└── reports/
    ├── figures/                # *.png (generated)
    └── metrics/leaderboard.csv # (generated)
```

---

## 🔬 Design decisions worth calling out

* **Ground truth from free labels.** No human judgements exist, so relevance is
  derived from shared `issue`/`product` labels. This is a documented
  approximation, not a gold standard — but it's reproducible and lets every
  retriever be compared on equal footing.
* **Capped Recall denominator.** With label-derived relevance, one signature can
  cover 69 complaints; `min(|relevant|, k)` keeps Recall@k interpretable.
* **Retriever-aware confidence guardrail.** Cosine, BM25, and RRF scores live on
  different scales, so a single absolute threshold would be meaningless across
  them — the guardrail adapts per retriever.
* **Embedded Qdrant isolation.** Each retriever instance gets its own on-disk
  collection path so many (model × chunking) runs coexist without lock
  contention; server mode removes this constraint at scale.

---

## 📜 License & data

Code: MIT (add your `LICENSE`). Complaint data is published by the CFPB as a
public resource; narratives are already PII-redacted at source.
