"""
Ring-Puzzle-Solver — FastAPI Backend
"""

import asyncio
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from solver import load_wordlist, build_ring, find_valid_words, solve, check_word, get_tracking_table


# --- Global State ---
raw_words: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global raw_words
    t0 = time.time()
    raw_words = load_wordlist()
    dt = time.time() - t0
    print(f"Wordlist loaded: {len(raw_words)} words in {dt:.1f}s")
    yield


app = FastAPI(title="Ring-Puzzle-Solver", lifespan=lifespan)

ALLOWED_ORIGINS = [
    "https://ring.solvingsystems.de",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- Request / Response Models ---

class SolveRequest(BaseModel):
    letters: str
    max_words: int = 2

    @field_validator("letters")
    @classmethod
    def validate_letters(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-ZÄÖÜ]{12}$", v):
            raise ValueError("Genau 12 Buchstaben (A-Z) erforderlich")
        return v

    @field_validator("max_words")
    @classmethod
    def validate_max_words(cls, v: int) -> int:
        if v < 1 or v > 4:
            raise ValueError("max_words muss zwischen 1 und 4 liegen")
        return v


class CheckRequest(BaseModel):
    letters: str
    word: str

    @field_validator("letters")
    @classmethod
    def validate_letters(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-ZÄÖÜ]{12}$", v):
            raise ValueError("Genau 12 Buchstaben (A-Z) erforderlich")
        return v

    @field_validator("word")
    @classmethod
    def validate_word(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) < 2:
            raise ValueError("Wort muss mindestens 2 Buchstaben haben")
        return v


# --- API Endpoints ---

@app.get("/api/health")
async def health():
    return {"status": "ok", "words_loaded": len(raw_words)}


@app.post("/api/solve")
async def api_solve(req: SolveRequest):
    ring_data = build_ring(req.letters)

    def _solve():
        words = find_valid_words(ring_data, raw_words)
        solutions = solve(ring_data, words, max_words=req.max_words)
        return words, solutions

    try:
        words, solutions = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _solve),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Solver-Timeout (10s)")

    # Tracking fuer jede Loesung hinzufuegen
    for sol in solutions:
        sol["tracking"] = []
        for word in sol["chain"]:
            sol["tracking"].append(get_tracking_table(word, ring_data))

    return {
        "letters": req.letters,
        "ring": ring_data["ring"],
        "neighbors": {k: sorted(v) for k, v in ring_data["neighbors"].items()},
        "valid_words_count": len(words),
        "solutions": solutions,
    }


@app.post("/api/check")
async def api_check(req: CheckRequest):
    ring_data = build_ring(req.letters)
    result = check_word(ring_data, req.word)
    return result


# --- Static Files (Frontend) ---
app.mount("/", StaticFiles(directory="public", html=True), name="static")
