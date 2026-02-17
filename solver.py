"""
Ring-agnostischer Buchstabenraetsel-Solver
==========================================
Importierbares Modul fuer beliebige 12-Buchstaben-Ringe.
"""

import gzip
import os
from typing import Optional


# --- Wortqualitaets-Daten (einmal laden, global cachen) ---

_hunspell_lemmas: Optional[set[str]] = None
_word_frequencies: Optional[dict[str, int]] = None


def _get_data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_hunspell_lemmas() -> set[str]:
    """Laedt Hunspell-Lemmata als Whitelist (uppercase)."""
    global _hunspell_lemmas
    if _hunspell_lemmas is not None:
        return _hunspell_lemmas

    _hunspell_lemmas = set()
    for ext in [".txt.gz", ".txt"]:
        filepath = os.path.join(_get_data_dir(), f"hunspell_lemmas{ext}")
        if not os.path.exists(filepath):
            continue
        opener = gzip.open if ext.endswith(".gz") else open
        with opener(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    _hunspell_lemmas.add(word.upper())
        break
    return _hunspell_lemmas


def load_word_frequencies() -> dict[str, int]:
    """Laedt Wortfrequenzen (uppercase -> frequency)."""
    global _word_frequencies
    if _word_frequencies is not None:
        return _word_frequencies

    _word_frequencies = {}
    for ext in [".txt.gz", ".txt"]:
        filepath = os.path.join(_get_data_dir(), f"word_frequencies{ext}")
        if not os.path.exists(filepath):
            continue
        opener = gzip.open if ext.endswith(".gz") else open
        with opener(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].upper()
                    try:
                        freq = int(parts[1])
                    except ValueError:
                        continue
                    _word_frequencies[word] = freq
        break
    return _word_frequencies


def _is_known_compound(word: str, hunspell: set[str]) -> bool:
    """Prueft ob ein Wort ein Kompositum aus bekannten Teilen ist."""
    w = word.upper()
    # Versuche alle Splits: mindestens 3 Buchstaben pro Teil
    for i in range(3, len(w) - 2):
        left = w[:i]
        right = w[i:]
        if left in hunspell and right in hunspell:
            return True
        # Fugen-S: NASENBEIN → NASEN+BEIN (NASEN als Plural in Hunspell)
        if left.endswith("S") and left[:-1] in hunspell and right in hunspell:
            return True
        # Fugen-N: SONNENSCHEIN → SONNEN+SCHEIN
        if left.endswith("N") and left[:-1] in hunspell and right in hunspell:
            return True
        # Fugen-EN: STRAHLENBÜNDEL → STRAHLEN+BÜNDEL
        if left.endswith("EN") and left[:-2] in hunspell and right in hunspell:
            return True
    return False


def word_quality_score(word: str) -> tuple[int, int, int]:
    """
    Sortier-Score fuer ein Wort. Niedrigere Werte = bessere Qualitaet.
    Returns (tier, neg_frequency, length):
      tier 0 = Hunspell + haeufig
      tier 1 = Hunspell oder haeufig (Freq > 50) oder bekanntes Kompositum
      tier 2 = In Frequenzliste (aber selten)
      tier 3 = Weder Hunspell noch Frequenz (unbekannt)
    """
    hunspell = load_hunspell_lemmas()
    freqs = load_word_frequencies()

    w = word.upper()
    in_hunspell = w in hunspell
    freq = freqs.get(w, 0)

    if in_hunspell and freq > 50:
        tier = 0
    elif in_hunspell or freq > 50:
        tier = 1
    elif freq > 0 or _is_known_compound(w, hunspell):
        tier = 2
    else:
        tier = 3

    return (tier, -freq, len(w))


def build_ring(letters: str) -> dict:
    """
    Baut Ring-Datenstruktur aus einem 12-Buchstaben-String.
    Gibt dict mit 'ring', 'ring_set', 'neighbors' zurueck.
    """
    ring = [ch.upper() for ch in letters]
    if len(ring) != 12:
        raise ValueError(f"Ring muss genau 12 Buchstaben haben, bekam {len(ring)}")

    n = len(ring)
    neighbors = {}
    for i, ch in enumerate(ring):
        left = ring[(i - 1) % n]
        right = ring[(i + 1) % n]
        neighbors[ch] = {left, ch, right}

    return {
        "ring": ring,
        "ring_set": set(ring),
        "neighbors": neighbors,
    }


def is_valid_word_in_ring(word: str, ring_data: dict) -> bool:
    """Prueft ob ein Wort die Ring-Sperr-Regel erfuellt."""
    word = word.upper()
    ring_set = ring_data["ring_set"]
    neighbors = ring_data["neighbors"]

    if not all(ch in ring_set for ch in word):
        return False

    for i in range(len(word) - 1):
        current = word[i]
        next_ch = word[i + 1]
        if next_ch in neighbors[current]:
            return False

    return True


def get_tracking_table(word: str, ring_data: dict) -> list[dict]:
    """Erzeugt Tracking-Schema fuer ein Wort (Schritt, Buchstabe, Gesperrt, Verfuegbar)."""
    word = word.upper()
    ring_set = ring_data["ring_set"]
    neighbors = ring_data["neighbors"]
    table = []
    for i, ch in enumerate(word):
        blocked = neighbors[ch]
        available = ring_set - blocked
        table.append({
            "step": i + 1,
            "letter": ch,
            "blocked": sorted(blocked),
            "available": sorted(available),
            "next": word[i + 1] if i + 1 < len(word) else None,
        })
    return table


def check_word(ring_data: dict, word: str) -> dict:
    """Einzelwort-Validierung mit Tracking-Info."""
    word = word.upper()
    ring_set = ring_data["ring_set"]
    neighbors = ring_data["neighbors"]

    missing_letters = [ch for ch in word if ch not in ring_set]
    if missing_letters:
        return {
            "valid": False,
            "reason": "letters_not_in_ring",
            "missing": missing_letters,
            "tracking": [],
        }

    valid = is_valid_word_in_ring(word, ring_data)
    tracking = get_tracking_table(word, ring_data)

    violation = None
    if not valid:
        for row in tracking:
            if row["next"] and row["next"] in set(row["blocked"]):
                violation = {
                    "step": row["step"],
                    "letter": row["letter"],
                    "blocked": row["blocked"],
                    "next": row["next"],
                }
                break

    covered = list(set(word) & ring_set)

    return {
        "valid": valid,
        "reason": None if valid else "ring_rule_violation",
        "violation": violation,
        "word": word,
        "covered": sorted(covered),
        "coverage": len(covered),
        "tracking": tracking,
    }


# --- Wortlisten-Verwaltung ---

def is_likely_base_form(word: str, all_words: set[str]) -> bool:
    """Heuristik: Filtert typische deutsche Beugungsformen heraus."""
    w = word.upper()

    for suffix, base_suffix in [
        ("BARE", "BAR"), ("BAREN", "BAR"), ("BARER", "BAR"),
        ("BARES", "BAR"), ("BAREM", "BAR"),
    ]:
        if w.endswith(suffix) and w[:-len(suffix)] + base_suffix in all_words:
            return False

    for suffix in [
        "ERE", "EREN", "ERER", "EREM", "ERES",
        "ERNE", "ERNEN", "ERNER", "ERNEM",
        "NAHE", "NAHEN", "NAHER", "NAHEM",
    ]:
        if w.endswith(suffix):
            for base_len in [len(suffix) - 1, len(suffix) - 2]:
                if base_len > 0 and w[:-base_len] in all_words:
                    return False

    if w.endswith("E") and len(w) > 4:
        infinitiv = w[:-1] + "EN"
        if infinitiv in all_words and infinitiv != w:
            return False

    for suffix in ["ENE", "ENEN", "ENER", "ENEM"]:
        if w.endswith(suffix) and w[:-len(suffix)] + "EN" in all_words:
            return False

    # Genitiv/Plural-S: BECKENBAUERS→BECKENBAUER, KAISERS→KAISER etc.
    if w.endswith("S") and len(w) > 5 and w[:-1] in all_words:
        return False

    # -ES Genitiv: KREISES→KREIS, ALBERNES→ALBERN
    if w.endswith("ES") and len(w) > 5 and w[:-2] in all_words:
        return False

    # Adjektiv-Deklination: NEBELREICHER→NEBELREICH, UNREINEN→UNREIN
    for suffix in ["ER", "EN", "EM"]:
        if w.endswith(suffix) and len(w) > 6 and w[:-len(suffix)] in all_words:
            base = w[:-len(suffix)]
            # Nur wenn Basis ein Adjektiv sein koennte (nicht KINDER→KIND etc.)
            # Pruefe: Basis + andere Adj-Endungen existieren auch
            if (base + "ER" in all_words or base + "EN" in all_words
                    or base + "ES" in all_words or base + "E" in all_words):
                return False

    # Verb-Konjugation 3. Person: -T → Infinitiv -EN
    if w.endswith("T") and len(w) > 5:
        # Regulaer: ZANKT→ZANKEN, TANKT→TANKEN
        if w[:-1] + "EN" in all_words:
            return False
        # Starke Verben mit I→E Ablaut: ZERBRICHT→ZERBRECHEN, SPRICHT→SPRECHEN
        stem = w[:-1]
        if "I" in stem:
            idx = stem.rfind("I")
            alt_stem = stem[:idx] + "E" + stem[idx + 1:]
            if alt_stem + "EN" in all_words:
                return False

    # Kontrahierte Formen: GLUCKRE→GLUCKERN, WANDRE→WANDERN
    if w.endswith("RE") and len(w) > 5 and w[:-2] + "ERN" in all_words:
        return False

    return True


def load_wordlist(filepath: Optional[str] = None) -> set[str]:
    """
    Laedt Wortliste aus (ggf. gzip-komprimierter) Datei.
    Gibt alle Rohwoerter (uppercase) zurueck — Ring-Filterung passiert spaeter.
    """
    raw_words: set[str] = set()

    if filepath is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        filepath = os.path.join(data_dir, "german_words.txt.gz")

    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip().split()[0] if line.strip() else ""
                word = word.strip('.,;:!?"\'()[]{}')
                if word:
                    raw_words.add(word.upper())
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip().split()[0] if line.strip() else ""
                word = word.strip('.,;:!?"\'()[]{}')
                if word:
                    raw_words.add(word.upper())

    return raw_words


def find_valid_words(ring_data: dict, raw_words: set[str], apply_filter: bool = True) -> set[str]:
    """Filtert Wortliste fuer einen bestimmten Ring."""
    ring_set = ring_data["ring_set"]

    valid = set()
    for word in raw_words:
        if len(word) < 4:
            continue
        if not all(ch in ring_set for ch in word):
            continue
        if is_valid_word_in_ring(word, ring_data):
            valid.add(word)

    if apply_filter:
        # Pruefe gegen volle Wortliste, nicht nur ring-valide Woerter
        # (Infinitiv ZERBRECHEN muss gefunden werden, auch wenn nicht ring-valid)
        valid = {w for w in valid if is_likely_base_form(w, raw_words)}

    return valid


def letters_used(word: str, ring_set: set[str]) -> set[str]:
    """Welche Ring-Buchstaben werden in einem Wort verwendet?"""
    return set(word.upper()) & ring_set


def solve(ring_data: dict, words: set[str], max_words: int = 2) -> list[dict]:
    """
    Backtracking-Solver. Gibt Liste von Loesungen zurueck.
    Jede Loesung: {"chain": [...], "covered": [...], "coverage": int}
    Sammelt alle 12/12-Loesungen (max 500) und sortiert nach Gesamtlaenge
    (kuerzere Woerter = gebraeuchlicher).
    """
    ring = ring_data["ring"]
    ring_set = ring_data["ring_set"]

    by_start: dict[str, list[str]] = {}
    for w in words:
        by_start.setdefault(w[0], []).append(w)

    # Sortiere nach Qualitaet: Hunspell+haeufig > Hunspell > Rest, dann Laenge
    for key in by_start:
        by_start[key].sort(key=word_quality_score)

    perfect_solutions: list[list[str]] = []
    best_imperfect: list[list[str]] = []
    best_coverage = 0
    max_perfect = 500

    def backtrack(chain: list[str], covered: set[str], depth: int):
        nonlocal best_imperfect, best_coverage

        coverage = len(covered)

        if coverage == 12 and chain:
            perfect_solutions.append(list(chain))
            return

        if depth >= max_words:
            if coverage > best_coverage:
                best_coverage = coverage
                best_imperfect = [list(chain)]
            elif coverage == best_coverage and chain:
                best_imperfect.append(list(chain))
            return

        if chain:
            next_start = chain[-1][-1]
            if next_start not in by_start:
                return

            for word in by_start[next_start]:
                new_letters = letters_used(word, ring_set) - covered
                if not new_letters and coverage < 12:
                    continue

                chain.append(word)
                backtrack(chain, covered | letters_used(word, ring_set), depth + 1)
                chain.pop()

                if len(perfect_solutions) >= max_perfect:
                    return
        else:
            for start_letter in ring:
                if start_letter not in by_start:
                    continue
                for word in by_start[start_letter]:
                    chain.append(word)
                    backtrack(chain, covered | letters_used(word, ring_set), depth + 1)
                    chain.pop()
                    if len(perfect_solutions) >= max_perfect:
                        return

    backtrack([], set(), 0)

    # Perfekte Loesungen bevorzugen, nach Gesamtlaenge sortieren
    solutions = perfect_solutions if perfect_solutions else best_imperfect

    # Sortiere nach Gesamtqualitaet: Summe der Tier-Scores, dann Frequenz
    def chain_quality(chain):
        scores = [word_quality_score(w) for w in chain]
        total_tier = sum(s[0] for s in scores)
        total_neg_freq = sum(s[1] for s in scores)
        total_len = sum(s[2] for s in scores)
        return (total_tier, total_neg_freq, total_len)

    solutions.sort(key=chain_quality)

    results = []
    for chain in solutions:
        covered = set()
        for w in chain:
            covered |= letters_used(w, ring_set)
        results.append({
            "chain": chain,
            "covered": sorted(covered),
            "coverage": len(covered),
        })

    return results
