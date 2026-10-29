# MedQuery

**An Intelligent Medical Question Answering System Using BERT and Knowledge Retrieval**

MedQuery takes a medical question in plain language, processes it with a pretrained
BERT model, retrieves evidence from a connected medical knowledge base, and extracts
an answer only when that evidence supports one. Every answer is shown next to the
passage it came from and that passage's source.

> **Educational prototype.** MedQuery does not diagnose, treat or prescribe, and its
> output is not verified medical advice.

> **The knowledge base is built locally, not committed.** `python -m knowledge.build all`
> downloads MedlinePlus, MedQuAD, a BioASQ subset and PubMed abstracts and indexes them
> (about 45,000 documents). Until it has been built and enabled in `.env`, every query says
> that evidence retrieval is unavailable. It never shows a made-up answer or citation.
> See [Built-in knowledge base](#built-in-knowledge-base).

---

## Contents

1. [Architecture](#architecture)
2. [Technology stack](#technology-stack)
3. [Setup and running](#setup-and-running)
4. [What BERT does in MedQuery](#what-bert-does-in-medquery)
5. [API reference](#api-reference)
6. [Answer statuses](#answer-statuses)
7. [Configuration](#configuration)
8. [Built-in knowledge base](#built-in-knowledge-base)
9. [Connecting your knowledge base](#connecting-your-knowledge-base)
10. [Multilingual questions and voice (Sarvam AI)](#multilingual-questions-and-voice-sarvam-ai)
11. [Evaluation](#evaluation)
12. [Testing](#testing)
13. [Known limitations](#known-limitations)

---

## Architecture

```
            ┌──────────────────────────── backend/ (FastAPI) ─────────────────────────────┐
 Browser    │                                                                              │
 frontend/ ─┼─► POST /api/query ─► answer_processing/pipeline.py                           │
            │                        │                                                     │
            │                        ├─1─► query_processing/                               │
            │                        │       preprocess.py   clean, normalise, keywords,   │
            │                        │                       question type, safety flags   │
            │                        │       bert_encoder.py BERT WordPiece tokens +       │
            │                        │                       384-d question embedding      │
            │                        │                                                     │
            │                        ├─2─► retrieval/service.py  timeout, errors, filtering│
            │                        │       └─► adapters/medquery_kb.py: the BERT        │
            │                        │           embedding searched with FAISS, then      │
            │                        │           re-checked in SQLite (or your adapter)   │
            │                        │                                                     │
            │                        ├─3─► EVIDENCE GATE: no usable passages → no answer   │
            │                        │                                                     │
            │                        └─4─► answer_processing/generators.py                 │
            │                                extractive BERT-QA span │ evidence-only       │
            │                                                                              │
            │   app/storage.py  SQLite query history  (GET /api/query/{id}, /api/history)  │
            └──────────────────────────────────────────────────────────────────────────────┘
```

```
medquery/
├── .env.example                 configuration template (no secrets)
├── README.md
├── frontend/                    React + Tailwind chat UI (Vite); dist/ is served by the backend
│   ├── src/App.jsx              conversation state, API calls
│   ├── src/components/          sidebar, header, messages, composer
│   ├── src/lib/                 REST client, pure helpers, constants
│   ├── dist/                    production build (`npm run build`)
│   └── tests/                   Vitest unit + React interaction tests
└── backend/
    ├── requirements.txt         web app + test dependencies
    ├── requirements-ml.txt      torch, transformers, sentence-transformers, faiss-cpu (optional)
    ├── pytest.ini
    ├── app/                     FastAPI app, config, routes, storage, model loading
    ├── query_processing/        preprocessing + BERT encoder
    ├── knowledge/               built-in knowledge base: schema, importers, chunking, index, CLI
    ├── retrieval/               interface, NOT_CONFIGURED adapter, adapter template
    │   └── adapters/            medquery_kb.py (built-in KB) + template for your own
    ├── answer_processing/       evidence gate, answer generators, response models
    ├── multilingual/            Sarvam AI client, language registry, translate-in / translate-out
    ├── evaluation/              Exact Match / F1, retrieval accuracy, scope tests; results/
    └── tests/                   pytest suite (mocks only)
```

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI 0.115** + Uvicorn | Async framework with input validation and auto-generated docs at `/api/docs` |
| Validation / config | **Pydantic 2**, pydantic-settings | Typed request/response models; settings come from env vars or `.env` |
| NLP | **Hugging Face Transformers 4.46** + **PyTorch 2.5 (CPU)** | Standard way to load pretrained BERT; CPU wheels avoid a GPU requirement |
| Query encoder | `sentence-transformers/all-MiniLM-L6-v2` (configurable) | A 6-layer BERT trained for sentence similarity; its question embedding is what the knowledge base is searched with |
| Answer extraction | `deepset/bert-base-cased-squad2` (configurable) | BERT fine-tuned on SQuAD 2.0, which can also say "no answer here" |
| Storage | SQLite (Python standard library) | Query history and the knowledge base, with no extra service |
| Passage embeddings | the same `all-MiniLM-L6-v2`, via sentence-transformers 3.3 | Questions and passages must share one model for their vectors to be comparable |
| Vector search | **FAISS** 1.15 (`IndexFlatIP`, exact cosine) | Standard vector-search library; falls back to identical numpy search if `faiss-cpu` is missing |
| Frontend | React 18 + Tailwind CSS 3 + lucide-react, built with Vite 5 | Component UI; the build output is served by FastAPI |
| Evaluation | `backend/evaluation/` | Exact Match / F1 (SQuAD rules), Hit@k, Recall@k, MRR, refusal rate |
| Tests | pytest, Vitest, Testing Library, jsdom | Run offline, no knowledge base or paid API needed |

## Setup and running

Requirements: **Python 3.10+** (tested on 3.12), and **Node 18+** to build or test the frontend.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate                     # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
# Optional but recommended: BERT models, FAISS and dataset tools (CPU-only torch is ~200 MB)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt

copy ..\.env.example ..\.env               # optional; defaults work without it

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**. The backend serves both the UI and the API. The UI is a
React + Tailwind app built with Vite; after changing anything in `frontend/src`, rebuild it
with `cd frontend && npm install && npm run build` (or use `npm run dev` for live reload on
port 5173, which forwards `/api` to the backend). Interactive API docs are at **http://localhost:8000/api/docs**.

**First query:** model weights download from the Hugging Face Hub on first use and are
cached under `~/.cache/huggingface`: the query encoder is about 90 MB and the QA model
about 430 MB. With the knowledge base unconfigured, only the query encoder is loaded; the
QA model loads the first time retrieval returns evidence. Set `MEDQUERY_PRELOAD_MODELS=true`
to load both in the background at startup, or `MEDQUERY_HF_OFFLINE=true` to use cached
weights only.

**Without torch/transformers** the app still runs. The encoder reports `unavailable`, queries
use keyword-only processing, and when evidence is present the app shows it without
extracting an answer (`EVIDENCE_ONLY`).

## What BERT does in MedQuery

BERT (Bidirectional Encoder Representations from Transformers) is used in two separate places:

1. **Query understanding and retrieval** (`query_processing/bert_encoder.py`). The question is
   split into WordPiece tokens (`[CLS] what are the common symptoms of ... [SEP]`) and passed
   through a pretrained BERT encoder (`all-MiniLM-L6-v2`, a 6-layer `BertModel`). Each token
   gets a 384-dimensional vector that depends on the words around it. MedQuery mean-pools those
   vectors (using the attention mask) into one normalised **question embedding**. The built-in
   knowledge base was indexed with the same model, so this embedding is searched directly with
   FAISS: the response's `retrieval.message` reads "Searched with bert query embedding (faiss)".
   If you configure a different encoder, the adapter notices the model mismatch and embeds the
   question with the index's model instead.
2. **Answer extraction** (`answer_processing/generators.py`). **BioBERT** (BERT pre-trained on
   PubMed, then fine-tuned for reading comprehension on SQuAD 2.0;
   `ktrapeznikov/biobert_v1.1_pubmed_squad_v2`) reads the question with each retrieved passage and
   predicts the start and end of an answer span, with a confidence score. The answer is
   copied word for word from the evidence, so it cannot add claims that aren't in your
   knowledge base. The SQuAD 2.0 model can also predict "no answer", which MedQuery reports
   as `NO_ANSWER_IN_EVIDENCE`. It was chosen over general BERT on measured results; see
   [Evaluation](#evaluation).

**What BERT does not do:** it is not a source of medical knowledge and it does not check
whether anything is medically correct. Without a knowledge base, MedQuery gives no answers at all.

## API reference

Every response uses one envelope:

```jsonc
{ "success": true,  "data": { ... } }
{ "success": false, "error": { "code": "validation_error", "message": "...", "details": [...] } }
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | App status, model load states, knowledge-base configuration, input limits |
| `POST` | `/api/query` | Run the full pipeline. Body: `{"question": "...", "language": "en-IN"}` (`language` optional) |
| `GET` | `/api/languages` | Supported languages, what each Sarvam service supports, and whether multilingual is configured |
| `POST` | `/api/transcribe` | Multipart `file` (audio, ≤ 30 s, ≤ 5 MB) + `language` → `{transcript}` via Sarvam speech-to-text |
| `POST` | `/api/speech` | `{"text": "...", "language": "kn-IN"}` → `audio/wav` via Sarvam text-to-speech |
| `GET` | `/api/query/{id}` | Fetch a stored result (`404` if unknown, `400` if the id is malformed) |
| `GET` | `/api/history` | Recent queries (`id`, `question`, `status`, `created_at`) |
| `DELETE` | `/api/history` | Delete all stored queries |
| `GET` | `/api/docs` | Swagger UI |

**Validation:** the question must be 3–500 characters after whitespace normalisation and
must contain letters. The request body is capped at 8 KB (`413 payload_too_large`). Invalid
input returns `422 validation_error`. Unexpected failures return `500 internal_error` with no
internal details exposed.

**`POST /api/query` response (`data`):**

```jsonc
{
  "id": "f0b9564c012749bcb2e5610115fa0e4d",
  "question": "What are the common symptoms of [condition]?",
  "status": "KB_NOT_CONFIGURED",          // see "Answer statuses"
  "answer": null,                          // or { text, confidence, method, evidence_index, verified:false }
  "evidence": [],                          // passages + used_for_answer, highlight_start/end
  "sources": [],                           // de-duplicated { source_id, title, url }
  "retrieval": { "status": "NOT_CONFIGURED", "adapter": "not_configured",
                 "message": "No medical knowledge base is connected...", "passages_returned": 0 },
  "analysis": {
    "normalized_text": "what are the common symptoms of [condition]?",
    "keywords": ["common", "symptoms", "condition"],
    "question_type": "symptoms",
    "safety_flags": [],
    "bert": { "available": true, "token_count": 12, "wordpieces": ["[CLS]", "what", "..."],
              "truncated": false, "embedding_dim": 384 }
  },
  "notices": [{ "level": "warning", "code": "kb_not_configured", "message": "..." }],
  "timings_ms": { "understanding": 94.1, "retrieval": 0.4, "total": 94.5 },
  "created_at": "2026-10-27T17:00:00Z",
  "disclaimer": "MedQuery is an educational prototype...",
  "localization": null                     // non-English questions only; see "Multilingual questions"
}
```

## Answer statuses

| Status | Meaning | Answer shown? |
|---|---|---|
| `ANSWERED` | Span extracted from evidence with confidence ≥ `QA_MIN_CONFIDENCE` | Yes, with evidence |
| `LOW_CONFIDENCE` | Span extracted, but confidence is below the threshold | Yes, with a warning |
| `NO_ANSWER_IN_EVIDENCE` | Evidence was found, but the QA model found no answer span in it | No; evidence shown |
| `EVIDENCE_ONLY` | Evidence was found; `evidence_only` mode, or the QA model is unavailable | No; evidence shown |
| `NO_EVIDENCE` | The knowledge base returned nothing relevant | No |
| `KB_NOT_CONFIGURED` | No knowledge base is connected | No |
| `RETRIEVAL_FAILED` | The adapter timed out or raised an error | No; UI offers retry |

Safety notices are added on top of the status: `possible_emergency` (urgent phrasing, so the user
is told to contact emergency services) and `personal_medical_advice` (requests for a personal
diagnosis or dose).

## Configuration

All settings are environment variables prefixed `MEDQUERY_`, read from the process environment
or from `.env` in the project root. See [.env.example](.env.example) for the full list with
comments. The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `MEDQUERY_QUERY_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Query encoder (any BERT-family model or local path); keep it equal to the index's model so its embedding drives the search |
| `MEDQUERY_KB_ADAPTER` | *(empty)* | `module.path:ClassName` of your knowledge-base adapter |
| `MEDQUERY_KB_CONNECTION` | *(empty)* | Connection string passed to your adapter |
| `MEDQUERY_KB_TOP_K` | `5` | Passages requested and kept |
| `MEDQUERY_KB_TIMEOUT_SECONDS` | `10` | Retrieval timeout |
| `MEDQUERY_KB_MIN_RELEVANCE` | `0.0` | Discard passages scoring below this |
| `MEDQUERY_ANSWER_MODE` | `extractive` | `extractive` or `evidence_only` |
| `MEDQUERY_QA_MODEL_NAME` | `deepset/bert-base-cased-squad2` | Extractive QA model |
| `MEDQUERY_QA_MIN_CONFIDENCE` | `0.15` | Below this, the status is `LOW_CONFIDENCE` |

No API keys are required. If you add an adapter that needs a secret, read it from an
environment variable inside the adapter and keep it in `.env`, which is git-ignored.

## Built-in knowledge base

### Sources

| Dataset | Documents | What it adds | Licence |
|---|---|---|---|
| [MedlinePlus health topics](https://medlineplus.gov/xml.html) | 1,016 | Patient-friendly summaries of conditions, symptoms and treatments (NLM) | Public domain |
| [MedQuAD](https://github.com/abachaa/MedQuAD) | 16,373 | Q&A pairs from 9 NIH sites, typed as symptoms, treatment, causes, etc. | CC BY 4.0 |
| [BioASQ subset](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) | 28,001 | PubMed abstracts from the BioASQ biomedical QA benchmark | CC BY 2.5 |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) via NCBI E-utilities | 495 | Recent review articles for 10 project topics (diabetes, hypertension, asthma, dengue, malaria, tuberculosis, anemia, migraine, kidney disease, heart disease) | NLM terms of use |

Not included:
- **MIMIC-III.** It requires credentialed PhysioNet access, and its data use agreement forbids
  sharing it with online services. Its notes are also patient records, not reference material.
- **Three MedQuAD subsets** (A.D.A.M. encyclopedia, drugs, herbs and supplements). The
  dataset's authors removed their answers for copyright reasons.

### Building it

```powershell
cd backend
.venv\Scripts\activate
pip install -r requirements-ml.txt
python -m knowledge.build all      # download, import, index (about 2 hours on a laptop CPU, mostly indexing)
```

Everything is written to `backend/data/kb/` (git-ignored). Each step can also be run on its
own: `download`, `import`, `dedupe`, `index` and `stats`. `dedupe` removes duplicate documents
from a knowledge base built before duplicate detection existed, keeping a backup of the
database and deleting the matching vectors without re-embedding anything. `--sources medlineplus,medquad` limits the build
to some datasets. Then enable it in `.env` and restart the server:

```ini
MEDQUERY_KB_ADAPTER=retrieval.adapters.medquery_kb:MedQueryKnowledgeBase
MEDQUERY_KB_MIN_RELEVANCE=0.45
```

### How it works

1. **Import** (`knowledge/sources.py`). Each dataset is read into documents that carry their
   source URL, publisher and licence. Records without text or a source URL are rejected.
   **Duplicates are skipped:** a document whose text matches one already stored (ignoring case,
   punctuation and spacing) is not imported again. MedQuAD republishes many MedlinePlus pages
   word for word; the knowledge base held 747 such copies before this check was added.
2. **Clean** (`knowledge/chunking.py`). HTML is converted to headings and paragraphs, and a
   short, explicit list of non-informative sentences is removed: attribution lines, NCI site
   navigation, and the paragraph GARD repeats in about 2,200 answers explaining its symptom
   table. Sentences that repeat but carry facts (such as inheritance patterns) are kept.
3. **Chunk.** Passages of about 180 words with a 30-word overlap. They never cross a section
   heading, and the heading is kept at the top of the passage.
4. **Store** (`knowledge/db.py`). SQLite with the five tables from the project design:
   `medical_sources`, `medical_topics`, `medical_documents`, `document_chunks` and `medical_claims`.
5. **Index** (`knowledge/index.py`). Each approved passage, prefixed with its document title, is
   embedded and L2-normalised, and the vectors are loaded into a FAISS `IndexFlatIP` (exact
   inner-product search, which equals cosine similarity on normalised vectors).
6. **Search** (`retrieval/adapters/medquery_kb.py`). The pipeline's BERT question embedding is
   searched with FAISS, and the top passages by cosine similarity are re-checked in SQLite. Only passages whose
   document **and** source are still `approved` are returned.

### Measured results

Retrieval accuracy, answer Exact Match / F1 and the relevance threshold are measured by
`backend/evaluation/`; see [Evaluation](#evaluation). Index build time is about 2 hours on a
laptop CPU, and a search takes a few milliseconds.

### Review status

Imported documents are `approved` at **source level**: they come from NIH/NLM or peer-reviewed
literature, but nobody has read each one. This is not the per-document expert review the
project design describes. Change the status of a document or a whole dataset with:

```powershell
python -m knowledge.build review --document 42 --status rejected
python -m knowledge.build review --dataset bioasq --status pending
```

Rejecting or un-approving takes effect on the next query, with no rebuild needed. Approving
something new requires `python -m knowledge.build index`. The `medical_claims` table is
created but left empty, ready for claim-level review.

## Connecting your knowledge base

The built-in knowledge base is one adapter. To use a different one, MedQuery only needs one
class that implements `retrieval.base.Retriever`.

### 1. The contract (`backend/retrieval/base.py`)

**Input: `RetrievalQuery`**

| Field | Type | Notes |
|---|---|---|
| `original_text` | `str` | Cleaned question, original casing |
| `normalized_text` | `str` | Lower-cased, contractions expanded |
| `keywords` | `list[str]` | Stop-words removed; useful for BM25 / keyword search |
| `question_type` | `str` | `definition`, `symptoms`, `causes`, `treatment`, `prevention`, `diagnosis`, `prognosis`, `comparison`, `general` |
| `embedding` | `list[float] \| None` | L2-normalised BERT embedding (768-d for base models); `None` if BERT is unavailable |
| `top_k` | `int` | How many passages to return |

**Output: `RetrievalResult`**

| Field | Type | Notes |
|---|---|---|
| `status` | `RetrievalStatus` | `OK`, `NO_RESULTS` or `ERROR` (`TIMEOUT` and `NOT_CONFIGURED` are set by MedQuery) |
| `passages` | `list[Passage]` | Must be empty unless status is `OK` |
| `message` | `str \| None` | Optional explanation shown to the user |

**`Passage`:** `text` (required, used for answer extraction), `source_id` (required),
`source_title`, `source_url` (http/https only are rendered as links), `relevance_score`
(higher is better; 0–1 recommended), `metadata` (free-form dict).

**Rules:** return only real knowledge-base content; return `NO_RESULTS` when nothing matches;
raise an exception on failure (MedQuery turns it into `RETRIEVAL_FAILED`). `retrieve()` runs in
a worker thread with a timeout, so blocking I/O is fine.

### 2. Implement the adapter

Edit [backend/retrieval/adapters/custom_adapter_template.py](backend/retrieval/adapters/custom_adapter_template.py)
(or copy it to a new module in the same folder) and fill in `__init__` (open your index or
database from `self.connection`), `retrieve`, and optionally `close`.

### 3. Point MedQuery at it (`.env`)

```ini
MEDQUERY_KB_ADAPTER=retrieval.adapters.custom_adapter_template:MyKnowledgeBaseAdapter
MEDQUERY_KB_CONNECTION=path/or/url/to/your/knowledge/base
```

Restart the server. `GET /api/health` should now show
`"knowledge_base": {"configured": true, "adapter": "my_knowledge_base", ...}` and the status pill
in the UI turns green. If the adapter fails to import, the app still starts, logs the error,
and reports the knowledge base as not configured, with the reason.

### 4. Tune

- If your scores are on a 0–1 scale, set `MEDQUERY_KB_MIN_RELEVANCE` to filter weak matches.
- If your passages are long, split them into chunks of roughly 100–300 words. The QA model reads
  up to 384 tokens at a time, so shorter passages give better answer spans.
- If you use dense retrieval, embed your documents with the **same** model as
  `MEDQUERY_QUERY_MODEL_NAME` so the vectors are comparable. Mean-pool over the attention mask
  and L2-normalise, as `bert_encoder.py` does.
- If your knowledge base is biomedical text, try `MEDQUERY_QUERY_MODEL_NAME=dmis-lab/biobert-base-cased-v1.1`.
- To show evidence without extracting answers, set `MEDQUERY_ANSWER_MODE=evidence_only`.

### Files you may need to edit

| File | Why |
|---|---|
| `backend/retrieval/adapters/custom_adapter_template.py` (or a new file next to it) | **Your adapter. Usually the only code change.** |
| `.env` (copied from `.env.example`) | `MEDQUERY_KB_ADAPTER`, `MEDQUERY_KB_CONNECTION`, tuning values |
| `backend/requirements.txt` | Add any client library your knowledge base needs (e.g. `faiss-cpu`, `psycopg`, `elasticsearch`) |
| `backend/tests/` | Add tests for your adapter |
| `backend/app/config.py` | Only if your adapter needs extra typed settings |

## Multilingual questions and voice (Sarvam AI)

Questions can be typed or spoken in **English, Hindi, Kannada, Tamil or Telugu**, and the answer
comes back in the selected language, with an optional "Listen to answer" button. Translation
and speech run on [Sarvam AI](https://docs.sarvam.ai); the medical answer still comes only
from the knowledge base and BioBERT.

```
question (kn/hi/ta/te)  ──Sarvam translate (mayura:v1)──►  English question
voice question ──Sarvam speech-to-text (saaras:v4)──► transcript ──► user reviews/edits ──┘
                                                                                          │
                 existing pipeline: BERT embedding → FAISS → approved passages → BioBERT span
                                                                                          │
answer, status message, notices  ◄──Sarvam translate──  English answer + evidence + sources
        └── "Listen to answer" ──Sarvam text-to-speech (bulbul:v3)──► WAV in the browser
```

### What is translated, and what is not

- **Translated:** the question (to English), the extracted answer span, the status message
  ("No reliable information found", "treat with caution", ...) and safety notices, including the
  emergency notice.
- **Kept in English, always shown:** the English answer ("English original: ..."), the English
  question that was searched ("Searched in English for ..."), and all evidence passages and
  sources. A translation can therefore always be checked against the evidence.
- **Never done:** translation never produces an answer on its own. If retrieval finds no
  evidence, the user gets the (translated) "no reliable information" message, not a
  machine-generated answer.
- **Numbers are checked.** Translations use international numerals, and if any number in the
  English answer (a dose, age, duration) is missing from the translation, the answer shows
  "Check the numbers ... rely on the English text" (`warnings: ["numbers_changed"]`).
- **Failures are explicit.** If the question can't be translated, the request fails with a clear
  message instead of searching with a guess. If part of the reply can't be translated, that
  part stays in English and the reply says "Partly in English" (`partial_translation`).
- **No unnecessary translation.** English questions never call Sarvam. A Latin-script question
  with another language selected is sent with `source_language_code: "auto"`; if Sarvam
  detects English, the original text is searched unchanged.

### Getting a Sarvam API key

1. Sign up at **https://dashboard.sarvam.ai** and verify your account (verified accounts get
   ₹100 of free credits).
2. Open **API Keys**, create a key and copy it. The dashboard shows it only once.
3. Put **only the key** in `.env` in the project root (git-ignored), with nothing else on the line:

   ```ini
   SARVAM_API_KEY=sk_your_key_here
   ```
4. Restart the server. `GET /api/languages` should report `"multilingual_configured": true`.

Without a key MedQuery runs in English only: the other languages are greyed out in the menu
("needs Sarvam key"), and the microphone falls back to the browser's built-in speech
recognition. An invalid key is reported as "The Sarvam AI API key is invalid" and English
keeps working.

### Install and run

```powershell
cd backend
.venv\Scripts\activate
pip install -r requirements.txt          # adds python-multipart (uploads) and httpx (Sarvam calls)
uvicorn app.main:app --port 8000
```

The frontend build in `frontend/dist` already includes the language menu and voice controls;
after changing `frontend/src`, run `cd frontend && npm install && npm run build`.

### Trying it

- **Typed question:** pick ಕನ್ನಡ (or हिन्दी, தமிழ், తెలుగు) in the language menu next to the
  microphone, type e.g. `ರಕ್ತಹೀನತೆಯ ಸಾಮಾನ್ಯ ಲಕ್ಷಣಗಳು ಯಾವುವು?` and send. The reply shows the
  Kannada answer, "English original", "Searched in English for ..." and the English sources.
- **Voice question:** press the microphone and allow access, speak (up to 29 seconds; a timer
  shows how long), press stop. The transcript appears in the box for you to check and edit;
  nothing is sent until you press send.
- **Listen to answer:** press it under any reply to hear it in the selected language; press
  again to stop, and again to replay. Audio is only generated when you press it.
- **Errors:** a blocked microphone, an unsupported browser, a recording that is too short,
  noise with no speech, timeouts and an invalid key each show their own message.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SARVAM_API_KEY` | *(empty)* | Sarvam key; empty = English only |
| `MEDQUERY_SARVAM_STT_MODEL` | `saaras:v4` | Speech-to-text model |
| `MEDQUERY_SARVAM_TRANSLATE_MODEL` | `mayura:v1` | Translation model |
| `MEDQUERY_SARVAM_TTS_MODEL` | `bulbul:v3` | Text-to-speech model |
| `MEDQUERY_SARVAM_TTS_SPEAKER` | *(Sarvam default, `shubh`)* | A bulbul:v3 voice, lowercase (e.g. `ritu`, `priya`, `kavya`) |
| `MEDQUERY_SARVAM_TIMEOUT_SECONDS` | `20` | Per-request timeout |

**Adding a language:** add one line to `LANGUAGES` in `backend/multilingual/languages.py`
(code, English name, native name, and which services support it). Every language on the list
below is supported by all three models, so e.g. `Language("ml-IN", "Malayalam", "മലയാളം")` is
enough; the menu picks it up from `/api/languages`.

### Sarvam limits and costs (checked on docs.sarvam.ai, October 2026)

| Service | Model | Languages | Per request | Price (pay as you go, excl. GST) | Rate limit (pay as you go) |
|---|---|---|---|---|---|
| Speech-to-text | `saaras:v4` | 22 Indian + English | ≤ 30 s of audio (REST) | ₹0.49 / minute, billed per second | 60 / min |
| Translation | `mayura:v1` | en + 10 Indian (hi bn ta te gu kn ml mr pa od) | ≤ 1,000 characters | ₹2 / 1,000 characters | 60 / min |
| Text-to-speech | `bulbul:v3` | the same 11 | ≤ 2,500 characters | ₹3 / 1,000 characters | 60 / min |

Rate limits are per account. A typical non-English question makes 3–5 translation calls (the
question, the answer, the status message and any notices), so one user asking about one
question a minute stays far below them. Pro and Growth plans raise the limits; the free ₹100
credit covers roughly 50,000 translated characters. Prices change; check
https://docs.sarvam.ai/api/getting-started/pricing.

### Privacy

The key stays on the server and is never sent to the browser or written to logs. Audio and
question text are sent to Sarvam AI for processing (see Sarvam's own terms), but MedQuery
never logs them: Sarvam failures are logged by status and error code only.

## Evaluation

`backend/evaluation/evaluate.py` measures the system on public question sets with gold answers.
No model is trained in MedQuery, so there is no training split: each set is split once,
deterministically by a hash of the question id, into **dev** (used to choose models and
thresholds) and **test** (held out, used only for the numbers below). Every run writes
`backend/evaluation/results/<task>-<split>.json` with per-question rows and a `-failures.csv`
listing the questions the system got wrong, for error analysis.

```powershell
cd backend
.venv\Scriptsctivate
python -m evaluation.evaluate retrieval  --split test              # BioASQ: Hit@k, Recall@k, MRR@10
python -m evaluation.evaluate reader     --split test              # COVID-QA: Exact Match and F1
python -m evaluation.evaluate end-to-end --split test --limit 300  # BioASQ through the whole pipeline
python -m evaluation.evaluate scope                                # answer in-domain, refuse off-topic
python -m evaluation.evaluate reader --split dev --model dmis-lab/biobert-base-cased-v1.1-squad
```

Results below were measured on 29 Oct 2026 (laptop CPU, knowledge base of 45,138 documents /
77,889 passages after de-duplication).

### Answer extraction: Exact Match and F1

[COVID-QA](https://huggingface.co/datasets/deepset/covid_qa_deepset) (Apache 2.0): 2,019 questions
written by biomedical experts on COVID-19 papers, each with the exact answer span. The model is
given the paragraph that contains the answer and must extract it. Exact Match and F1 follow the
official SQuAD script (case, punctuation and articles ignored; F1 over tokens).

Model selection on **dev** (first 400 questions):

| QA model | Exact Match | F1 | No answer given |
|---|---|---|---|
| `deepset/bert-base-cased-squad2` (previous default) | 0.323 | 0.524 | 24.3 % |
| **`ktrapeznikov/biobert_v1.1_pubmed_squad_v2`** (BioBERT, SQuAD 2.0) | **0.370** | **0.606** | 9.5 % |
| `dmis-lab/biobert-base-cased-v1.1-squad` (BioBERT, SQuAD 1.1) | 0.320 | 0.571 | 10.8 % |

Final result on **test** (822 held-out questions):

| QA model | Exact Match | F1 |
|---|---|---|
| **BioBERT SQuAD 2.0 (now the default)** | **0.381** | **0.606** |
| General BERT SQuAD 2.0 (previous) | 0.293 | 0.497 |

### Retrieval accuracy

[BioASQ](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) (CC BY 2.5): 4,719
biomedical questions written by experts, each with the PubMed abstracts that answer it. The
whole knowledge base is searched (all four datasets), and a hit means a gold abstract was
retrieved. **Test** split, 2,368 questions:

| | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| Before de-duplication | 0.522 | 0.711 | 0.280 | 0.361 | 0.602 |
| **After de-duplication (current)** | **0.524** | **0.713** | **0.280** | **0.361** | **0.604** |

Hit@k: a gold abstract is in the top k. Recall@k: the share of all gold abstracts found in the
top k (questions have up to 30 gold abstracts, so this is necessarily lower). MRR@10: mean of
1 / rank of the first gold abstract.

### End to end

The first 300 BioASQ **test** questions, run through the whole pipeline exactly as the app does
(BERT embedding, FAISS, relevance threshold 0.45, BioBERT answer extraction, confidence check):

| Measure | BioBERT SQuAD 2.0 (current) |
|---|---|
| Questions answered (`ANSWERED` or `LOW_CONFIDENCE`) | 0.813 (197 answered, 47 low confidence) |
| Declined (`NO_ANSWER_IN_EVIDENCE` 45, `NO_EVIDENCE` 11) | 0.187 |
| Shown evidence includes a gold abstract | 0.750 |
| Answer extracted from a gold abstract (of answered questions) | 0.631 |
| Token F1 of the answer against the expert's ideal answer | 0.106 |

Fewer than one answer in five is declined; those are the cases where the system says it
found no supported answer rather than guessing.

### Answering versus refusing

`evaluation/data/scope_questions.json`: 30 everyday medical questions and 30 off-topic ones
(cooking, sport, travel, IT...), labelled by hand.

| Threshold (`MEDQUERY_KB_MIN_RELEVANCE`) | Medical questions answered | Off-topic questions refused |
|---|---|---|
| **0.45 (current)** | **30 / 30** | **30 / 30** |

Top relevance scores were 0.60–0.83 for medical questions and 0.19–0.39 for off-topic ones;
every threshold from 0.40 to 0.60 separates them perfectly, and 0.45 sits inside that range.
The set is small, so recheck the threshold if the knowledge base changes.

### Limitations of these numbers

- COVID-QA covers one disease and research-paper language; patient questions about other
  conditions may score differently.
- BioASQ gold answers are abstracts from PubMed; questions are often answered by MedlinePlus
  or MedQuAD passages too, which count as misses, so retrieval accuracy is a lower bound.
- The end-to-end F1 compares a short extracted span with a long expert summary, so it is low
  by construction; "answer from a gold passage" is the more meaningful end-to-end measure.

## Testing

```powershell
# Backend: 111 tests, all mocked: no model download, knowledge base, network or Sarvam key needed
cd backend
.venv\Scripts\python -m pytest

# Frontend: 26 tests (pure helpers + React interaction tests in jsdom, via Vitest)
cd frontend
npm install
npm test
```

What the tests cover: query preprocessing; request validation and error envelopes (422, 413,
404, 405, 500); the BERT encoder with a mocked tokenizer and model (lazy loading, pooling,
truncation, graceful failure); the `NOT_CONFIGURED` adapter; adapter loading from a dotted
path; timeouts and adapter exceptions; empty and filtered retrieval results; the pipeline with
and without evidence (and a check that the answer generator is never called without evidence);
low-confidence and no-answer cases; the extractive generator's span selection; persistence and
history; the knowledge base (HTML cleaning, chunking and overlap, boilerplate removal, rejected
records, approved-only retrieval, and rejection taking effect without re-indexing); duplicate
detection and the `dedupe` index pruning; FAISS and numpy search agreeing; reuse of the BERT
question embedding for search; model loads never overlapping; the evaluation metrics and
splits; and UI behaviour (status pill, templates, validation, result rendering, HTML escaping,
`javascript:` URL blocking, error/retry, and history replay).

**Multilingual and voice** (`backend/tests/test_multilingual.py`, `frontend/tests/multilingual.test.jsx`)
use a fake Sarvam server, so they need no key or network. They check the documented request
fields and `api-subscription-key` header; error mapping for invalid keys, quota, rate limits,
timeouts and server errors (without echoing Sarvam's message); English questions never calling
Sarvam; Kannada questions searching in English and coming back translated with evidence and
sources intact; auto-detection of Latin-script input; flagged number changes; the
no-evidence message in Tamil; partial translation falling back to English; question-translation
failure as an explicit error; English-only mode without a key; unsupported languages; upload
validation (empty, too large, wrong type, no speech); and in the browser: the language menu,
recording and transcript review (no auto-submit), a blocked microphone, transcription errors,
and listen / stop / replay.

The test doubles live in `backend/tests/doubles.py` and the frontend DOM test. They use neutral
placeholder strings, not medical content, and are never loaded by the running app.

## Known limitations

- **The knowledge base must be built locally** (about 2 hours on CPU) and enabled in `.env`. Until
  then every query returns `KB_NOT_CONFIGURED`.
- **Approval is at source level,** not per-document expert review (see [Review status](#review-status)).
- **Coverage is uneven.** MedQuAD leans towards rare and genetic diseases, BioASQ towards
  research questions, and PubMed covers only 10 topics. Common questions are best served by
  MedlinePlus, which has about 1,000 topics.
- **Similarity is not correctness.** The relevance threshold filters weak matches, but a
  passage that is close in meaning can still be the wrong evidence.
- **Extractive answers are short spans.** The QA model copies phrases from the evidence; it
  doesn't write summaries. Questions that need several facts combined may get
  `NO_ANSWER_IN_EVIDENCE` or a partial span.
- **Confidence is the model's own score,** not a measure of medical correctness. `verified` is always `false`.
- **Question-type and safety detection are keyword rules**, not a trained classifier. They can
  miss unusual phrasing or flag harmless questions.
- **CPU inference:** about 100 ms per query for the encoder on a laptop CPU. Extractive QA adds
  roughly 100–300 ms per passage. The first query is slow while weights download and load.
- **No login and no rate limiting.** History is shared by everyone using the server. This is
  fine for a local demo but not for public deployment.
- **Translation quality is Sarvam's.** MedQuery shows the English answer and evidence next to
  every translation and flags changed numbers, but it cannot verify that a translated sentence
  means exactly the same thing, and drug names may be transliterated. The interface itself
  (buttons, labels, error messages) is in English.
- **Retrieval runs in English.** A non-English question is only as good as its translation;
  the English text that was searched is shown so mistranslations are visible.
- **Voice questions are limited to 30 seconds** (Sarvam's real-time limit) and need a browser
  with `MediaRecorder` (current Chrome, Edge, Firefox and Safari). Without a Sarvam key, the
  browser's own speech recognition is used, which works in Chrome and Edge only.
- **No generative LLM.** Answer generation sits behind the `AnswerGenerator` interface in
  `answer_processing/generators.py`, so an optional LLM generator could be added later
  without changing the rest of the pipeline.
