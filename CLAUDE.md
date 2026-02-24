# Ring Puzzle Solver

Web app for solving letter ring puzzles (12 letters in a ring).

## Architecture

- **Backend:** FastAPI (`server.py`) + solver module (`solver.py`)
- **Frontend:** Vanilla JS/HTML/CSS (`public/`)
- **Word list:** `data/german_words.txt.gz` (~6MB, 1.9M words)
- **Quality data:** `data/hunspell_lemmas.txt.gz` (744k) + `data/word_frequencies.txt.gz` (50k)
- **Deploy:** Docker + Caddy on VPS (`ring.solvingsystems.de`)

## Development

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8080
open http://localhost:8080
```

## API

```
GET  /api/health  → {"status": "ok"}
POST /api/solve   → {letters: "RBEMPALZINHY", max_words: 2}
POST /api/check   → {letters: "RBEMPALZINHY", word: "PRIMZAHL"}
```

## Deploy

```bash
deploy ring-puzzle-solver --first-run
```
