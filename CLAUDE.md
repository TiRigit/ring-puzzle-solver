# Ring-Puzzle-Solver

Web-App zum Loesen von Buchstabenring-Raetseln (12 Buchstaben im Ring).

## Architektur

- **Backend:** FastAPI (`server.py`) + Solver-Modul (`solver.py`)
- **Frontend:** Vanilla JS/HTML/CSS (`public/`)
- **Wortliste:** `data/german_words.txt.gz` (~6MB, 1.9M Woerter)
- **Qualitaetsdaten:** `data/hunspell_lemmas.txt.gz` (744k) + `data/word_frequencies.txt.gz` (50k)
- **Deploy:** Docker + Caddy auf VPS (`ring.solvingsystems.de`)

## Entwicklung

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
