# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
make install        # Create venv and install dependencies
./setup.sh          # Alternative setup script

# Run API
make run            # Production mode
make dev            # Development mode (debug logging + auto-reload)
./run_api.sh        # Production via shell script
./run_dev.sh        # Development via shell script

# Manual run (from api/scripts/)
source ../../venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Database
make check-db                        # Verify database exists
python api/scripts/sdo_database.py   # Repopulate database from SciX (requires SCIX_API_KEY in .env)

# Docker
docker build -t nasa-ads-sdo-api .
docker run -p 8000:8000 -v $(pwd)/api/database:/app/api/database:ro nasa-ads-sdo-api

# Cleanup
make clean          # Remove venv and cache files
```

## Architecture

This is a FastAPI service exposing a pre-populated SQLite database of Solar Dynamics Observatory (SDO) research papers sourced from SciX (scixplorer.org — the platform superseding NASA ADS's classic search API; same Solr-based query syntax and Bearer-token auth, just a different host).

### Data flow

1. **Database population** (`api/scripts/sdo_database.py`): A one-time/manual script that queries SciX for papers from 2010–2024 matching `bibgroup:SDO OR abstract:"solar dynamics observatory"` — the OR is there so ADS's own librarian-curated `bibgroup:SDO` bibliography contributes for free if it's ever populated, but it currently returns **zero** results live (verified 2026-07-27; unlike `bibgroup:HST`, which works exactly as documented), so the full phrase required verbatim in the abstract is what's actually doing the filtering. This is deliberately the full phrase, not the loose acronym "SDO" (which lets in papers that only cite SDO in passing, or unrelated fields where the acronym collides) — note phrase search requires *double* quotes in ADS's query syntax, single quotes do not enforce it. Further restricted to `property:refereed`, `database:astronomy`, `doctype:article`, and the Astronomy & Astrophysics journal (`bibstem:A&A` — open access, so PDF downloads are reliable, and empirically the best signal-to-noise source so far; ~503 papers match this query for 2010–2024). Widen beyond A&A later if its coverage proves insufficient. Results are batch-inserted into the SQLite database. Requires `SCIX_API_KEY` in `.env`. See the query's inline comments in `sdo_database.py` for the full empirical rationale.

2. **API server** (`api/scripts/main.py`): FastAPI app that serves the pre-populated database. All endpoints read from SQLite; there is no write path through the API. PDF downloads are proxied live from SciX's link gateway using `httpx` with browser-like headers and redirect following.

### Module layout

- `api/modules/models.py` — SQLModel models: `SDODocument` (ORM table), `SDODocumentPublic` (response model with computed `ads_url`). The `ads_url` field is not stored in the DB; it is computed from `bibcode` at response time.
- `api/modules/database.py` — SQLite engine pointing to `api/database/sdo_papers_2010_2024.db`
- `api/modules/config.py` — Environment variable configuration via `python-dotenv`
- `api/scripts/main.py` — All API routes; `sys.path` manipulation is used to import from `api/modules/` since the script runs from within `api/scripts/`

### Key routing detail

There is a path conflict to be aware of: `/documents/search/` is defined before `/documents/{document_id}` to prevent FastAPI from interpreting `"search"` as a document ID. Preserve this ordering when adding routes.

### PDF download behavior

Two endpoints handle PDF downloads:
- `GET /documents/{id}/download-pdf` (with optional `?source=arxiv|publisher`) — tries arXiv first, then publisher
- `GET /documents/{id}/download-pdf/{pdf_type}` — direct source selection via path parameter

Both use NASA ADS link gateway URLs (`EPRINT_PDF` for arXiv, `PUB_PDF` for publisher). PDF detection checks `Content-Type` and sniffs for `<html` in the first 1000 bytes to reject HTML error pages masquerading as PDFs.

### Configuration (`.env`)

```properties
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
SCIX_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///api/database/sdo_papers_2010_2024.db
```

### Interactive API docs

Available at `/docs` (Swagger UI) and `/redoc` when the server is running.
