import math
from typing import Any, Dict, List, Tuple

import streamlit as st
from chromadb import PersistentClient


def iter_collection_documents(collection, page_size: int = 500):
    """Iterate over a Chroma collection in pages, yielding (id, meta, doc)."""
    offset = 0
    while True:
        batch = collection.get(
            include=["metadatas", "documents"],
            limit=page_size,
            offset=offset,
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        metas = batch.get("metadatas") or []
        docs = batch.get("documents") or []
        for i, cid in enumerate(ids):
            yield cid, (metas[i] if i < len(metas) else {}), (docs[i] if i < len(docs) else "")
        offset += len(ids)


st.set_page_config(page_title="Chunks", page_icon="📚", layout="wide")
st.title("📚 Chunks en mémoire")
st.caption("Parcourir les passages indexés dans Chroma (pagination / filtre par source)")

col_top_a, col_top_b, col_top_c = st.columns([2, 1, 1])

with col_top_a:
    src_filter = st.text_input("Filtre source (contient)", value="")
with col_top_b:
    page_size = st.selectbox("Taille de page", options=[50, 100, 200], index=0)
with col_top_c:
    refresh = st.button("🔄 Rafraîchir")

persist_dir = st.session_state.get("persist_dir", "vectorstore")
collection_name = st.session_state.get("collection", "eau_docs")

client = PersistentClient(path=persist_dir)
try:
    collection = client.get_collection(collection_name)
except Exception:
    st.error("Collection introuvable. Lancez une indexation pour créer des chunks.")
    st.stop()

# Charger les enregistrements avec filtre optionnel
records: List[Dict[str, Any]] = []
if src_filter:
    f = src_filter.lower()
    for cid, meta, doc in iter_collection_documents(collection):
        src = meta.get("file_path") or meta.get("filename") or meta.get("source") or meta.get("id") or ""
        if f in str(src).lower():
            records.append({
                "id": str(cid),
                "source": str(src),
                "json_path": str(meta.get("json_path", "")),
                "text": doc or "",
            })
else:
    # Pas de filtre: on ne charge que la page courante via offset/limit plus bas
    pass

# Pagination
if src_filter:
    total = len(records)
    n_pages = max(1, math.ceil(total / page_size))
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size
    view = records[start:end]
    st.write(f"Total filtré: {total} | Pages: {n_pages} | Page courante: {page}")
else:
    # Sans filtre: s'appuyer sur Chroma pour offset/limit
    try:
        total = int(collection.count())
    except Exception:
        total = 0
    n_pages = max(1, math.ceil(max(0, total) / page_size))
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    offset = (page - 1) * page_size
    batch = collection.get(include=["metadatas", "documents"], limit=page_size, offset=offset)
    view = []
    ids = batch.get("ids") or []
    metas = batch.get("metadatas") or []
    docs = batch.get("documents") or []
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        src = meta.get("file_path") or meta.get("filename") or meta.get("source") or meta.get("id") or ""
        view.append({
            "id": str(cid),
            "source": str(src),
            "json_path": str(meta.get("json_path", "")),
            "text": doc or "",
        })
    st.write(f"Total: {total} | Pages: {n_pages} | Page courante: {page}")

# Affichage
if not view:
    st.info("Aucun chunk à afficher pour les paramètres sélectionnés.")
else:
    for rec in view:
        with st.expander(f"{rec['id']} — {rec['source']}"):
            if rec.get("json_path"):
                st.caption(f"json_path: {rec['json_path']}")
            st.write(rec["text"]) 

