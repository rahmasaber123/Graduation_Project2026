# NBIS — Fingerprint Identification Service

Production-ready FastAPI service + browser dashboard for the NBIS fingerprint
identification model. Loads your trained model, FAISS index, mapping,
threshold, and SQLite DB **once** at startup.

```
nbis_service/
├── backend/
│   ├── main.py           ← FastAPI app: /health, /identify, /identify-base64, /record, /stats
│   ├── model_loader.py   ← NBISSystem class, mirrors notebook inference
│   └── db.py             ← NBISDatabase — SQLite JOIN accessor
├── frontend/
│   └── index.html        ← single-file dashboard (CSS + JS inlined)
├── artifacts/            ← PUT YOUR 6 FILES HERE (5 model artifacts + nbis.db)
│   └── README.md
├── requirements.txt
└── README.md
```

## What's new in v2.1

- **Full DB record on match.** `/identify` now returns the joined child + father + mother + hospital record from `nbis.db`, alongside the biometric match. Frontend renders it in four clearly-separated sections.
- **Single-file frontend.** All HTML, CSS, and JS live in `frontend/index.html`. Easier to edit, easier to deploy — no separate asset loading.
- **White theme with dark-blue accents** (`--navy: #0a2540`). Pro/clinical look rather than the dark console look.
- **Graceful degradation.** If `nbis.db` is missing the API still works — it just returns the biometric match without the joined record, and the frontend indicates `DB ✗` in the header.

## Prerequisites

- **Python 3.10+** (3.10 or 3.11 recommended)
- The 5 artifact files + `nbis.db` from your notebook

## Setup

```bash
# 1. Create a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy artifacts + DB into ./artifacts/
cp /path/to/your/nbis_artifacts_v2/*.keras  artifacts/
cp /path/to/your/nbis_artifacts_v2/*.index  artifacts/
cp /path/to/your/nbis_artifacts_v2/*.pkl    artifacts/
cp /path/to/your/nbis_artifacts_v2/*.json   artifacts/
cp /path/to/your/nbis_database/nbis.db      artifacts/
```

## Run

```bash
cd backend
python main.py
```

Or with uvicorn directly:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Then open: **http://localhost:8000**

## API

| Method | Path                  | Body                    | Response |
|--------|-----------------------|-------------------------|----------|
| GET    | `/health`             | —                       | System status + DB availability |
| GET    | `/stats`              | —                       | Session counters |
| GET    | `/record/{subject_id}`| —                       | JOIN record from `nbis.db` |
| POST   | `/identify`           | multipart `file=<image>`| Identification + joined DB record |
| POST   | `/identify-base64`    | `{"image": "<base64>"}` | Same as `/identify` |

### Example response (MATCH)

```json
{
  "status": "MATCH",
  "confidence": 97.57,
  "similarity": 0.9757,
  "threshold": 0.7474,
  "subject_id": "190",
  "hand": "Left",
  "finger": "middle",
  "identity_id": "190_Left_middle",
  "top_k": [ ... ],
  "latency_ms": 68.4,
  "database": {
    "child":    { "name": "Tamer Omar Morsi", "gender": "M", "birth_datetime": "...", ... },
    "father":   { "name": "Omar Morsi",  "national_id": "...", "phone": "...", ... },
    "mother":   { "name": "Rahma Mansour", "national_id": "...", "phone": "...", ... },
    "hospital": { "name": "Tanta University Hospital", "city": "Tanta", ... }
  }
}
```

If `nbis.db` is missing, `"database"` will be `null` instead — the biometric
match is still returned.

## Configuration (environment variables)

| Variable                | Default                          | Purpose |
|-------------------------|----------------------------------|---------|
| `NBIS_ARTIFACTS_DIR`    | `../artifacts`                   | Where the 5 model artifact files live |
| `NBIS_DB_PATH`          | `../artifacts/nbis.db`           | SQLite DB path |
| `NBIS_TOP_K`            | `5`                              | Default top-K for `/identify` |
| `NBIS_MAX_UPLOAD_MB`    | `10`                             | Max accepted upload size |

## Editing the frontend

The entire UI is in `frontend/index.html` — styles, markup, and JS in one
file. Common edits:

- **Change theme color.** Search for `--navy:` near the top of `<style>` and
  change the hex. Every accent uses CSS variables.
- **Add fields to a record section.** Find e.g. `<!-- Father record -->`,
  add a new `<div class="kv">` block with a unique id, then add a
  `txt("newId", f.fieldName)` call in the `renderResult` function.
- **Adjust what toasts fire.** Search for `toast(` in the `<script>`.

## Deployment notes

- **Single worker** (`--workers 1`) keeps TF + FAISS memory sane on CPU.
- **CORS** is open for local dev; restrict `allow_origins` in `backend/main.py`
  before exposing externally.
- **No authentication.** The DB record contains synthetic PII (names, phones,
  national IDs). Add an auth middleware before this touches a network.
- **Startup takes ~2–4 seconds** — TF graph warm-up runs automatically during
  `load()` so the first real request is fast.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/health` returns `{"ready": false, "load_error": ...}` | Confirm all 5 artifact files exist in `artifacts/` with exact names |
| Header shows `DB ✗` | `nbis.db` not found — copy it from `/content/nbis_database/` into `artifacts/`, or set `NBIS_DB_PATH` |
| `ValueError: Unknown layer: L2Normalize` | `model_loader.py` registers it via `@register_keras_serializable`; ensure it's being imported |
| `404 No record for subject_id=X` | The match's `subject_id` isn't in your DB — usually means you trained on more subjects than you generated DB rows for |
| Frontend shows "Offline" dot | Backend down, or CORS blocked the `/health` call. Check browser devtools. |
