import streamlit as st
from app.loader import load_documents
from app.indexer import build_or_load_index
from app.rag_engine import ask_question
import os

# ===============================
# CONFIG GLOBALE
# ===============================
st.set_page_config(page_title="IA Traitement de l’Eau", page_icon="💧", layout="wide")

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
LLM_NAME = "mistral"
EMB_NAME = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

st.title("💧 IA Technique — Traitement de l’Eau")
st.caption("Assistant local propulsé par LlamaIndex + Ollama (Mistral)")

# ===============================
# COLONNES INTERFACE
# ===============================
col1, col2 = st.columns([1, 2])

# ===============================
# CHARGEMENT & INDEXATION
# ===============================
with col1:
    st.subheader("📁 Données")
    st.write("Ce panneau charge les fichiers présents dans le dossier `data/`.")

    if st.button("📥 Charger & indexer"):
        with st.spinner("Lecture et indexation des documents..."):
            docs = load_documents(DATA_DIR)

            # Comptage rapide
            n_docs = len(docs)
            st.info(f"{n_docs} documents détectés.")

            if n_docs == 0:
                st.warning("Aucun fichier trouvé dans `data/`.")
            else:
                try:
                    index = build_or_load_index(
                        data_documents=docs,
                        persist_dir=VECTOR_DIR,
                        llm_name=LLM_NAME,
                        embedding_name=EMB_NAME,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    st.success(f"✅ Index créé ou mis à jour ({n_docs} documents).")
                except Exception as e:
                    st.error(f"Erreur lors de l’indexation : {e}")

    # Affiche le contenu du dossier data/
    if os.path.exists(DATA_DIR):
        files = []
        for root, _, fns in os.walk(DATA_DIR):
            for f in fns:
                files.append(os.path.join(root, f))
        if files:
            st.write(f"📂 Fichiers détectés ({len(files)}) :")
            st.code("\n".join(files[:15]))
        else:
            st.write("Aucun fichier dans `data/`.")

# ===============================
# MOTEUR DE RECHERCHE RAG
# ===============================
with col2:
    st.subheader("🔍 Recherche / Question")
    st.write("Pose une question sur ton corpus (PDF, Word, Excel, JSON…).")

    question = st.text_input("💭 Ta question :")

    if question:
        with st.spinner("Génération de la réponse..."):
            try:
                index = build_or_load_index(
                    data_documents=[],  # essaie de recharger
                    persist_dir=VECTOR_DIR,
                    llm_name=LLM_NAME,
                    embedding_name=EMB_NAME,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
            except Exception as e:
                st.error(
                    "⚠️ Aucun index existant détecté. "
                    "Clique d’abord sur **📥 Charger & indexer** après avoir ajouté des fichiers dans `data/`.\n\n"
                    f"Détail : {e}"
                )
                st.stop()

            # Requête IA
            try:
                answer, sources = ask_question(index, question, top_k=TOP_K, model_name=LLM_NAME)
                st.subheader("🧠 Réponse")
                st.write(answer)

                if sources:
                    st.subheader("📚 Sources (extraits)")
                    for s in dict.fromkeys(sources):
                        st.code(str(s))
            except Exception as e:
                st.error(f"Erreur pendant la génération : {e}")

# ===============================
# BAS DE PAGE
# ===============================
st.markdown("---")
st.caption("© 2025 — IA Locale Traitement de l’Eau · LlamaIndex + Ollama + Streamlit")
