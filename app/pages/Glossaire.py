import json
from pathlib import Path
import streamlit as st

BASE_APP = Path(__file__).resolve().parent.parent  # app/
ABBR_DIR = BASE_APP / "abreviations"
ABBR_FILE = ABBR_DIR / "abreviations.json"
REGEX_FILE = ABBR_DIR / "abreviations_regex.json"


def load_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def save_json(path: Path, data) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Erreur d'écriture {path.name}: {e}")
        return False


st.set_page_config(page_title="Glossaire d'abréviations", page_icon="🗂️", layout="wide")
st.title("🗂️ Glossaire d'abréviations")
st.caption("Éditez les abréviations utilisées lors de l'indexation et des requêtes.")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Dictionnaire (clé → expansion)")
    abbr = load_json(ABBR_FILE, {})
    # Edition simple sous forme de JSON texte (robuste et rapide)
    abbr_text = st.text_area("abreviations.json", value=json.dumps(abbr, ensure_ascii=False, indent=2), height=420)
    if st.button("💾 Enregistrer le dictionnaire"):
        try:
            data = json.loads(abbr_text)
            if save_json(ABBR_FILE, data):
                st.success("Dictionnaire enregistré. Relancez l'indexation pour appliquer côté embeddings.")
        except Exception as e:
            st.error(f"JSON invalide: {e}")

with col_b:
    st.subheader("Règles regex (pattern/replacement/flags)")
    rules = load_json(REGEX_FILE, [])
    rules_text = st.text_area("abreviations_regex.json", value=json.dumps(rules, ensure_ascii=False, indent=2), height=420)
    if st.button("💾 Enregistrer les règles"):
        try:
            data = json.loads(rules_text)
            if save_json(REGEX_FILE, data):
                st.success("Règles enregistrées. Relancez l'indexation pour appliquer côté embeddings.")
        except Exception as e:
            st.error(f"JSON invalide: {e}")

st.markdown("---")
st.info("Astuce: l'expansion à la requête peut être désactivée dans l'écran principal.")

