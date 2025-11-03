import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple


@lru_cache(maxsize=1)
def _load_resources() -> Tuple[Dict[str, str], List[Tuple[re.Pattern, str]]]:
    base_app = Path(__file__).resolve().parent  # app/
    abbr_dir = base_app / "abreviations"
    # Dictionnaire d'abréviations
    abbr_path = abbr_dir / "abreviations.json"
    mapping: Dict[str, str] = {}
    if abbr_path.exists():
        try:
            mapping = json.loads(abbr_path.read_text(encoding="utf-8"))
        except Exception:
            mapping = {}
    # Règles regex optionnelles
    regex_path = abbr_dir / "abreviations_regex.json"
    rules: List[Tuple[re.Pattern, str]] = []
    if regex_path.exists():
        try:
            raw = json.loads(regex_path.read_text(encoding="utf-8"))
            for r in raw:
                patt = r.get("pattern")
                repl = r.get("replacement", "")
                flags = 0
                for f in r.get("flags", []) or []:
                    if f.upper() == "I":
                        flags |= re.IGNORECASE
                    if f.upper() == "M":
                        flags |= re.MULTILINE
                if patt:
                    # Fix JSON-escaped backspace (\b) -> regex word boundary \b
                    # In JSON, "\b" becomes a backspace char (U+0008) after decoding.
                    # Replace that control char with a literal backslash-b sequence for regex.
                    patt = patt.replace("\u0008", "\\b")
                    rules.append((re.compile(patt, flags), repl))
        except Exception:
            pass
    # Fallback par défaut si le JSON est invalide ou vide
    if not rules:
        try:
            rules = [
                (re.compile(r"\bSaaS\b", re.IGNORECASE), "Software as a Service (SaaS)"),
                (re.compile(r"\bSt\.?\b", re.IGNORECASE), "Saint"),
                (re.compile(r"\bSte\.?\b", re.IGNORECASE), "Sainte"),
                (re.compile(r"^(\s*(?:[-\u2022]\s*)?)R(\s*(?:[:\-\u2013\u2014]\s+))", re.MULTILINE), r"\1Poste de relevage (R)\2"),
                (re.compile(r"(?<!\()\bR\b(?!\))", re.IGNORECASE), "Poste de relevage (R)"),
            ]
        except Exception:
            rules = []
    return mapping, rules


def expand_abbreviations(text: str) -> str:
    if not text:
        return text
    mapping, rules = _load_resources()
    out = text
    # Remplacements par dictionnaire (ordonnés par longueur décroissante)
    for abbr in sorted(mapping.keys(), key=len, reverse=True):
        exp = mapping[abbr]
        # Bordures de mot, insensible à la casse; préserve l'abréviation rencontrée
        pattern = re.compile(rf"\b{re.escape(abbr)}\b", re.IGNORECASE)

        def _repl(m: re.Match) -> str:
            seen = m.group(0)
            # Eviter double expansion si déjà '(ABBR)' adjacent
            if re.search(rf"\(\s*{re.escape(seen)}\s*\)", out[max(0, m.start()-5):m.end()+5]):
                return seen
            return f"{exp} ({seen})"

        out = pattern.sub(_repl, out)

    # Règles regex spécifiques
    for patt, repl in rules:
        try:
            out = patt.sub(repl, out)
        except Exception:
            continue
    return out
