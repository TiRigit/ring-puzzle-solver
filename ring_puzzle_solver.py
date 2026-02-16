#!/usr/bin/env python3
"""
Ring-Buchstabenrätsel-Solver
=============================
12 Buchstaben im Kreis angeordnet. Bilde eine Kette von Wörtern (mind. 4 Buchstaben),
wobei:
- Nach Benutzung eines Buchstabens: er selbst + seine Nachbarn im Ring sind für den
  NÄCHSTEN Schritt gesperrt
- Nächstes Wort beginnt mit dem Endbuchstaben des vorherigen
- Ziel: alle 12 Buchstaben mindestens einmal verwenden (stets mit 2 Wörtern möglich)

Spielregeln (Original):
- Erlaubt: Duden-Stichwörter als Grundform (inkl. Eigennamen, Abkürzungen)
- NICHT erlaubt: Beugungsformen (Plurale, Deklinationen, Konjugationen, Imperative)
- Beispiele: PRIMZAHL ✅, PRIMZAHLEN ❌ (Plural), HEILBARE ❌ (dekliniert)

Verwendung:
    python ring_puzzle_solver.py                      # Standardrätsel lösen (max. 2 Wörter)
    python ring_puzzle_solver.py --wordlist pfad.txt  # Eigene Wortliste (mit Beugungsform-Filter)
    python ring_puzzle_solver.py --max-words 3        # Max. Wörter in Kette
    python ring_puzzle_solver.py --no-filter           # Beugungsform-Filter deaktivieren
    python ring_puzzle_solver.py --verbose             # Schritt-für-Schritt-Ausgabe
"""

import argparse
import sys
import os
from itertools import product as iter_product
from typing import Optional

# ============================================================
# RING-KONFIGURATION
# ============================================================
# Buchstaben im Kreis: R → B → E → M → P → A → L → Z → I → N → H → Y → (zurück zu R)
RING = ['R', 'B', 'E', 'M', 'P', 'A', 'L', 'Z', 'I', 'N', 'H', 'Y']

def build_neighbors(ring: list[str]) -> dict[str, set[str]]:
    """Erstellt Nachbarschafts-Map für den Ring."""
    n = len(ring)
    neighbors = {}
    for i, ch in enumerate(ring):
        left = ring[(i - 1) % n]
        right = ring[(i + 1) % n]
        neighbors[ch] = {left, ch, right}  # sich selbst + Nachbarn
    return neighbors

NEIGHBORS = build_neighbors(RING)
RING_SET = set(RING)

# ============================================================
# WORT-VALIDIERUNG IM RING
# ============================================================

def is_valid_word_in_ring(word: str) -> bool:
    """
    Prüft, ob ein Wort unter der Ring-Sperr-Regel gebildet werden kann.
    
    Regel: Nach Benutzung von Buchstabe X sind X und seine Ring-Nachbarn
    für den NÄCHSTEN Buchstaben gesperrt. Danach wird die Sperre neu
    berechnet basierend auf dem neuen Buchstaben.
    
    LLM-Tracking-Schema:
    Schritt | Buchstabe | Gesperrt danach     | Verfügbar
    1       | Z         | {L, Z, I}           | {R,B,E,M,P,A,N,H,Y}
    2       | A         | {P, A, L}           | {R,B,E,M,Z,I,N,H,Y}
    ...
    """
    word = word.upper()
    
    # Alle Buchstaben müssen im Ring sein
    if not all(ch in RING_SET for ch in word):
        return False
    
    for i in range(len(word) - 1):
        current = word[i]
        next_ch = word[i + 1]
        # Nach current sind current + Nachbarn gesperrt
        blocked = NEIGHBORS[current]
        if next_ch in blocked:
            return False
    
    return True

def get_tracking_table(word: str) -> list[dict]:
    """
    Erzeugt das LLM-Tracking-Schema für ein Wort.
    Zeigt pro Schritt: Buchstabe, was danach gesperrt ist, was verfügbar bleibt.
    """
    word = word.upper()
    table = []
    for i, ch in enumerate(word):
        blocked = NEIGHBORS[ch]
        available = RING_SET - blocked
        table.append({
            'schritt': i + 1,
            'buchstabe': ch,
            'gesperrt': sorted(blocked),
            'verfuegbar': sorted(available),
            'naechster': word[i + 1] if i + 1 < len(word) else '-'
        })
    return table

# ============================================================
# WORTLISTE
# ============================================================

# Eingebettete Liste: ~120 Duden-Stichwörter (Grundformen), die nur aus
# RING-Buchstaben bestehen und die Ring-Sperr-Regel erfüllen.
# NUR Grundformen: Nomen (Singular), Adjektive (undekliniert), Verben (Infinitiv),
# Adverbien, Eigennamen (Duden-Stichwort). KEINE Beugungsformen.
EMBEDDED_WORDS = [
    # --- Nomen (Singular) ---
    "AHLE", "BIER", "BLEI", "EHRE", "EILE", "ERLE",
    "HARN", "HARZ", "HEIM", "HELM", "HERZ", "HIRN",
    "LEIB", "LEIM", "LENZ", "LIRA", "MAHL", "NERZ",
    "PIER", "REIM", "ZAHL",
    "BAHRE", "BIRNE", "ENZYM", "LEHRE", "LEIER", "LEIHE",
    "LEPRA", "LIANE", "MIENE", "MIEZE", "PERLE", "PRAHM",
    "PRIEL", "REIHE", "ZEILE",
    "ARZNEI", "RANZEN",
    "ANLEIHE", "PRIMZAHL",
    # --- Adjektive (Grundform, undekliniert) ---
    "HEHR", "NAHE", "PRIM", "ZAHM",
    "NAHBAR", "HEILBAR", "LERNBAR", "ZAHLBAR", "PEILBAR",
    "PERIPHER", "BLEIERN", "HERZNAH",
    "ABNEHMBAR", "ERLERNBAR", "ABZAHLBAR", "ANZAHLBAR",
    # --- Verben (Infinitiv) ---
    "HEILEN", "LEHREN", "LEIERN", "LEIHEN", "MAHLEN",
    "PEILEN", "PRAHLEN", "ZAHLEN", "ZEHREN", "ZEIHEN",
    "HARZEN", "LENZEN", "LERNEN", "PIEPEN", "REIHEN",
    "EPILIEREN", "ARMIEREN",
    "ABHEILEN", "ABLEIERN", "ABLERNEN", "ABPERLEN",
    "ABZAHLEN", "ABZEHREN",
    "ANLEIERN", "ANLERNEN", "ANMAILEN", "ANPEILEN",
    "ANRANZEN", "ANREIHEN", "ANZAHLEN",
    "ERLERNEN", "HERLEIERN", "HERLEIHEN",
    "HERANEILEN", "REPRIMIEREN", "ZERMAHLEN", "HEIMZAHLEN",
    # --- Adverbien ---
    "HERAB", "HERAN", "HIER", "HIERAN", "HIERHER",
    # --- Eigennamen (Duden-Stichwörter) ---
    "IRAN", "LIMA", "BIRMA", "LIBYEN", "BAYERN", "RHEIN",
]

def is_likely_base_form(word: str, all_words: set[str]) -> bool:
    """
    Heuristik: Ist das Wort wahrscheinlich eine Grundform (Duden-Stichwort)?
    Filtert typische deutsche Beugungsformen heraus.
    """
    w = word.upper()

    # Adjektiv-Deklinationen: -bare/-baren/-barer/-bares von -bar
    for suffix, base_suffix in [("BARE", "BAR"), ("BAREN", "BAR"), ("BARER", "BAR"),
                                 ("BARES", "BAR"), ("BAREM", "BAR")]:
        if w.endswith(suffix) and w[:-len(suffix)] + base_suffix in all_words:
            return False

    # Adjektiv-Deklinationen: -e/-en/-er/-em/-es wenn Stamm existiert
    # (aber NICHT bei Verben auf -en, die behalten wir)
    for suffix in ["ERE", "EREN", "ERER", "EREM", "ERES",  # Komparativ-Dekl.
                    "ERNE", "ERNEN", "ERNER", "ERNEM",       # -ern Deklinationen
                    "NAHE", "NAHEN", "NAHER", "NAHEM"]:       # -nah Deklinationen
        if w.endswith(suffix):
            # Pruefe ob kuerzere Grundform existiert
            for base_len in [len(suffix) - 1, len(suffix) - 2]:
                if base_len > 0 and w[:-base_len] in all_words:
                    return False

    # Plurale auf -EN wo Singular existiert (PRIMZAHLEN->PRIMZAHL, ARZNEIEN->ARZNEI)
    if w.endswith("EN") and len(w) > 5:
        singular = w[:-2]
        if singular in all_words:
            # Aber: Verben auf -EN behalten (HEILEN ist Infinitiv, nicht Plural von HEIL)
            # Heuristik: wenn Singular ein Adjektiv/Nomen ist und das Wort KEIN typischer Infinitiv
            # -> eher Plural. Wir behalten es, wenn der Stamm nicht eigenstaendig ist.
            pass  # Komplexer Fall, wird unten behandelt

    # Konjugierte Verbformen (nicht Infinitiv)
    # Imperative / 1. Person: -E am Ende wenn Stamm+EN existiert
    if w.endswith("E") and len(w) > 4:
        infinitiv = w[:-1] + "EN"
        if infinitiv in all_words and infinitiv != w:
            return False  # z.B. PRAHLE -> PRAHLEN existiert

    # Praeteritum: NAHM, LIEH etc. (kurze Formen ohne -EN)
    # Schwer zu erkennen — wir lassen den Filter hier konservativ

    # Partizip-artige Formen: ZERMAHLENE, ZERMAHLENEN etc.
    for suffix in ["ENE", "ENEN", "ENER", "ENEM"]:
        if w.endswith(suffix) and w[:-len(suffix)] + "EN" in all_words:
            return False

    return True


def load_wordlist(filepath: Optional[str] = None, apply_filter: bool = True) -> set[str]:
    """
    Lädt Wortliste aus Datei oder nutzt eingebettete Liste.
    Filtert auf: nur Ring-Buchstaben, mind. 4 Zeichen, Ring-Regel gültig.
    Bei externer Liste: zusätzlicher Beugungsform-Filter (deaktivierbar mit --no-filter).
    """
    raw_words = set()
    from_external = False

    if filepath and os.path.exists(filepath):
        print(f"📖 Lade Wortliste aus: {filepath}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip().split()[0] if line.strip() else ""
                word = word.strip('.,;:!?"\'()[]{}')
                if word:
                    raw_words.add(word.upper())
        print(f"   {len(raw_words)} Rohwörter geladen")
        from_external = True
    else:
        if filepath:
            print(f"⚠️  Datei nicht gefunden: {filepath}")
        print("📖 Verwende eingebettete Wortliste (nur Grundformen)")
        raw_words = {w.upper() for w in EMBEDDED_WORDS}

    # Ring-Filterung
    valid = set()
    for word in raw_words:
        if len(word) < 4:
            continue
        if not all(ch in RING_SET for ch in word):
            continue
        if is_valid_word_in_ring(word):
            valid.add(word)

    print(f"   {len(valid)} Wörter nach Ring-Filterung")

    # Beugungsform-Filter bei externer Liste
    if from_external and apply_filter:
        before = len(valid)
        valid = {w for w in valid if is_likely_base_form(w, valid)}
        removed = before - len(valid)
        if removed > 0:
            print(f"   ⚙️  Beugungsform-Filter: {removed} Formen entfernt (--no-filter zum Deaktivieren)")

    print(f"✅ {len(valid)} gültige Wörter")
    return valid

# ============================================================
# SOLVER
# ============================================================

def letters_used(word: str) -> set[str]:
    """Welche Ring-Buchstaben werden in einem Wort verwendet?"""
    return set(word.upper()) & RING_SET

def chain_coverage(chain: list[str]) -> set[str]:
    """Welche Ring-Buchstaben deckt die gesamte Kette ab?"""
    covered = set()
    for w in chain:
        covered |= letters_used(w)
    return covered

def solve(wordlist: set[str], max_words: int = 4, verbose: bool = False) -> list[list[str]]:
    """
    Findet Wortketten, die möglichst alle 12 Buchstaben abdecken.
    
    Strategie: Backtracking mit Pruning.
    - Sortiert Wörter nach Buchstabenabdeckung (gierig)
    - Bricht ab, wenn max_words erreicht
    """
    # Index: Wörter nach Anfangsbuchstaben
    by_start: dict[str, list[str]] = {}
    for w in wordlist:
        start = w[0]
        by_start.setdefault(start, []).append(w)
    
    # Sortiere jede Gruppe: längere Wörter (mehr Abdeckung) zuerst
    for key in by_start:
        by_start[key].sort(key=lambda w: len(letters_used(w)), reverse=True)
    
    best_solutions: list[list[str]] = []
    best_coverage = 0
    
    def backtrack(chain: list[str], covered: set[str], depth: int):
        nonlocal best_solutions, best_coverage
        
        coverage = len(covered)
        
        # Neue beste Lösung?
        if coverage > best_coverage:
            best_coverage = coverage
            best_solutions = [list(chain)]
            if verbose:
                print(f"  🎯 Neue beste Abdeckung: {coverage}/12 mit {len(chain)} Wörtern: {' → '.join(chain)}")
                print(f"     Abgedeckt: {sorted(covered)}")
                print(f"     Fehlend:   {sorted(RING_SET - covered)}")
        elif coverage == best_coverage:
            best_solutions.append(list(chain))
        
        # Alle abgedeckt? Perfekt!
        if coverage == 12:
            return
        
        # Max Tiefe erreicht?
        if depth >= max_words:
            return
        
        # Nächster Anfangsbuchstabe = letzter Buchstabe des letzten Worts
        if chain:
            next_start = chain[-1][-1]
        else:
            # Erstes Wort: probiere alle Anfangsbuchstaben
            for start_letter in RING:
                if start_letter not in by_start:
                    continue
                for word in by_start[start_letter]:
                    new_covered = covered | letters_used(word)
                    chain.append(word)
                    backtrack(chain, new_covered, depth + 1)
                    chain.pop()
                    if best_coverage == 12 and len(best_solutions) >= 5:
                        return  # Genug perfekte Lösungen gefunden
            return
        
        next_start = chain[-1][-1]
        if next_start not in by_start:
            return
        
        for word in by_start[next_start]:
            # Pruning: Wort muss mindestens einen neuen Buchstaben bringen
            new_letters = letters_used(word) - covered
            if not new_letters and coverage < 12:
                continue
            
            new_covered = covered | letters_used(word)
            chain.append(word)
            backtrack(chain, new_covered, depth + 1)
            chain.pop()
            
            if best_coverage == 12 and len(best_solutions) >= 5:
                return
    
    print(f"\n🔍 Suche Wortketten (max. {max_words} Wörter)...")
    backtrack([], set(), 0)
    
    return best_solutions

# ============================================================
# AUSGABE
# ============================================================

def print_ring():
    """Zeigt den Ring visuell an."""
    print("\n🔵 RING-ANORDNUNG:")
    print(f"   {'  →  '.join(RING)}  → (zurück zu {RING[0]})")
    print()
    print("   Nachbarschaften (gesperrt nach Benutzung):")
    for ch in RING:
        neighbors = sorted(NEIGHBORS[ch])
        print(f"   {ch}: sperrt {neighbors}")

def print_solution(chain: list[str], index: int):
    """Gibt eine Lösung mit LLM-Tracking-Schema aus."""
    covered = chain_coverage(chain)
    missing = RING_SET - covered
    
    print(f"\n{'='*60}")
    print(f"LÖSUNG #{index}: {len(chain)} Wörter, {len(covered)}/12 Buchstaben")
    print(f"{'='*60}")
    print(f"Kette: {' → '.join(chain)}")
    print(f"Abgedeckt:  {sorted(covered)}")
    if missing:
        print(f"Fehlend:    {sorted(missing)}")
    else:
        print("✅ ALLE BUCHSTABEN ABGEDECKT!")
    
    # Tracking-Tabelle für LLM
    print(f"\n{'─'*60}")
    print("LLM-TRACKING-SCHEMA:")
    print(f"{'─'*60}")
    
    for wi, word in enumerate(chain):
        print(f"\nWort {wi+1}: {word}")
        table = get_tracking_table(word)
        print(f"  {'Schritt':<8} {'Buchst.':<9} {'Gesperrt danach':<20} {'Nächster':<9} {'OK?'}")
        for row in table:
            next_ch = row['naechster']
            blocked_str = ','.join(row['gesperrt'])
            if next_ch != '-':
                ok = '✅' if next_ch not in set(row['gesperrt']) else '❌'
            else:
                ok = '—'
            print(f"  {row['schritt']:<8} {row['buchstabe']:<9} {blocked_str:<20} {next_ch:<9} {ok}")
        
        if wi < len(chain) - 1:
            next_word = chain[wi + 1]
            print(f"  ➡️  Kette: '{word}' endet mit '{word[-1]}' → '{next_word}' beginnt mit '{next_word[0]}' ✅")

def print_llm_prompt_template(chain: list[str]):
    """
    Erzeugt ein Prompt-Template, das einer LLM das Mitrechnen ermöglicht.
    """
    print(f"\n{'='*60}")
    print("LLM-PROMPT-TEMPLATE FÜR DIESES RÄTSEL")
    print(f"{'='*60}")
    print("""
Du löst ein Buchstaben-Ring-Rätsel. Die Buchstaben stehen im Kreis:
R → B → E → M → P → A → L → Z → I → N → H → Y → (zurück zu R)

REGELN:
1. Bilde Wörter (mind. 4 Buchstaben) aus diesen Buchstaben
2. Nur Duden-Stichwörter als GRUNDFORM (Eigennamen + Abkürzungen erlaubt)
3. KEINE Beugungsformen (kein Plural, keine Deklination, keine Konjugation)
4. Nach Benutzung eines Buchstabens: er + seine Ring-Nachbarn sind
   für den NÄCHSTEN Buchstaben gesperrt
5. Nächstes Wort beginnt mit dem Endbuchstaben des vorherigen
6. Ziel: alle 12 Buchstaben mit genau 2 Wörtern abdecken (immer möglich)

TRACKING-FORMAT (nutze dies zum Mitrechnen):
Buchstabe | Gesperrt danach (Buchstabe + Nachbarn) | Nächster muss aus Restmenge kommen

NACHBARSCHAFTEN:""")
    for ch in RING:
        neighbors = sorted(NEIGHBORS[ch])
        print(f"  {ch}: sperrt {neighbors}")
    
    print(f"""
BEISPIEL-DURCHLAUF:
Wort: ZAHL
  Z → sperrt [I,L,Z] → nächster muss aus {{R,B,E,M,P,A,N,H,Y}} kommen
  A ✅ (nicht gesperrt) → sperrt [A,L,P] → nächster aus {{R,B,E,M,Z,I,N,H,Y}}
  H ✅ → sperrt [H,N,Y] → nächster aus {{R,B,E,M,P,A,L,Z,I}}
  L ✅ → Ende des Wortes
  
Nächstes Wort muss mit 'L' beginnen.
Bisher abgedeckt: {{Z,A,H,L}} — fehlend: {{R,B,E,M,P,I,N,Y}}
""")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ring-Buchstabenrätsel-Solver")
    parser.add_argument('--wordlist', '-w', type=str, default=None,
                       help='Pfad zur Wortliste (eine Wort pro Zeile)')
    parser.add_argument('--max-words', '-m', type=int, default=2,
                       help='Maximale Anzahl Wörter in der Kette (Standard: 2)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Beugungsform-Filter für externe Wortlisten deaktivieren')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Ausführliche Ausgabe während der Suche')
    parser.add_argument('--check', '-c', type=str, nargs='+',
                       help='Prüfe ob ein oder mehrere Wörter gültig sind')
    parser.add_argument('--track', '-t', type=str,
                       help='Zeige Tracking-Tabelle für ein Wort')
    parser.add_argument('--prompt', '-p', action='store_true',
                       help='Gib ein LLM-Prompt-Template aus')
    
    args = parser.parse_args()
    
    print_ring()
    
    # Nur Prompt-Template ausgeben
    if args.prompt:
        print_llm_prompt_template([])
        return
    
    # Einzelwort prüfen
    if args.check:
        for word in args.check:
            word = word.upper()
            valid = is_valid_word_in_ring(word)
            all_in_ring = all(ch in RING_SET for ch in word)
            print(f"\n🔎 Prüfe: {word}")
            if not all_in_ring:
                missing = [ch for ch in word if ch not in RING_SET]
                print(f"   ❌ Buchstaben nicht im Ring: {missing}")
            elif not valid:
                print(f"   ❌ Verletzt Ring-Sperr-Regel")
                table = get_tracking_table(word)
                for row in table:
                    next_ch = row['naechster']
                    if next_ch != '-' and next_ch in set(row['gesperrt']):
                        print(f"      Schritt {row['schritt']}: '{row['buchstabe']}' sperrt {row['gesperrt']}, "
                              f"aber nächster ist '{next_ch}' → GESPERRT!")
                        break
            else:
                print(f"   ✅ Gültig!")
                covered = letters_used(word)
                print(f"   Abgedeckt: {sorted(covered)} ({len(covered)}/12)")
            
            if valid or (all_in_ring and args.track is None):
                table = get_tracking_table(word)
                print(f"\n   {'Schritt':<8} {'Buchst.':<9} {'Gesperrt':<20} {'Nächster':<9}")
                for row in table:
                    blocked_str = ','.join(row['gesperrt'])
                    print(f"   {row['schritt']:<8} {row['buchstabe']:<9} {blocked_str:<20} {row['naechster']:<9}")
        return
    
    # Tracking für ein Wort
    if args.track:
        word = args.track.upper()
        if is_valid_word_in_ring(word):
            print(f"\n📊 Tracking für: {word}")
            table = get_tracking_table(word)
            print(f"   {'Schritt':<8} {'Buchst.':<9} {'Gesperrt danach':<20} {'Verfügbar':<30} {'Nächster':<9}")
            for row in table:
                blocked_str = ','.join(row['gesperrt'])
                avail_str = ','.join(row['verfuegbar'])
                print(f"   {row['schritt']:<8} {row['buchstabe']:<9} {blocked_str:<20} {avail_str:<30} {row['naechster']:<9}")
        else:
            print(f"❌ '{word}' ist kein gültiges Ring-Wort")
        return
    
    # Solver
    wordlist = load_wordlist(args.wordlist, apply_filter=not args.no_filter)
    
    if not wordlist:
        print("\n❌ Keine gültigen Wörter gefunden!")
        print("   Bitte lade eine deutsche Wortliste herunter und übergib sie mit --wordlist")
        print("   Empfehlung: https://github.com/davidak/wortliste")
        return
    
    # Zeige einige gültige Wörter
    print(f"\n📝 Beispiele gültiger Wörter:")
    sorted_words = sorted(wordlist, key=lambda w: len(letters_used(w)), reverse=True)
    for w in sorted_words[:15]:
        covered = letters_used(w)
        print(f"   {w:<15} deckt ab: {sorted(covered)} ({len(covered)} Buchstaben)")
    
    solutions = solve(wordlist, max_words=args.max_words, verbose=args.verbose)
    
    if solutions:
        # Zeige beste Lösungen
        print(f"\n🏆 {len(solutions)} Lösung(en) mit {len(chain_coverage(solutions[0]))}/12 Abdeckung gefunden:")
        for i, chain in enumerate(solutions[:5], 1):
            print_solution(chain, i)
    else:
        print("\n❌ Keine Lösung gefunden. Versuche eine größere Wortliste.")
    
    # Immer das Prompt-Template ausgeben
    print_llm_prompt_template(solutions[0] if solutions else [])

if __name__ == '__main__':
    main()
