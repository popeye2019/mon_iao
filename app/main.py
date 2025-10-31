import os
import streamlit as st

from loader import load_documents
from indexer import build_or_load_index
from rag_engine import ask_question
from utils.config import load_config


st.set_page_config(page_title="IA locale — Traitement de l'eau", layout="wide")
st.title("💧 IA technique — Traitement de l’eau (RAG local)")

cfg = load_config()

DATA_DIR = cfg["paths"]["data_dir"]
IMAGES_DIR = cfg["paths"]["images_dir"]
VECTOR_DIR = cfg["paths"]["vectorstore_dir"]
LLM_NAME = cfg["model"]["llm_name"]
EMB_NAME = cfg["model"]["embedding_name"]
TOP_K = cfg["indexing"]["top_k"]
CHUNK_SIZE = cfg["indexing"]["chunk_size"]
CHUNK_OVERLAP = cfg["indexing"]["chunk_overlap"]

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.text(f"LLM: {LLM_NAME}")
    st.text(f"Embeddings: {EMB_NAME}")
    st.text(f"Top-K: {TOP_K}")
    st.text(f"Chunks: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    st.markdown("---")
    st.text(f"Données: {DATA_DIR}")
    st.text(f"Images: {IMAGES_DIR}")
    st.text(f"Vectorstore: {VECTOR_DIR}")

st.markdown("""**Mode d'emploi**  
1. Place tes fichiers (PDF, DOCX, XLSX, JSON, TXT) dans le dossier `data/` (arborescence libre).  
2. Clique sur **Charger & indexer** pour créer/mettre à jour l'index.  
3. Pose tes questions. Les sources citées seront affichées si disponibles.
""")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("📥 Charger & indexer", type="primary"):
        with st.spinner("Chargement des documents..."):
            docs = load_documents(DATA_DIR)
            st.success(f"{len(docs)} documents chargés.")
        with st.spinner("Construction / rechargement de l'index..."):
            index = build_or_load_index(
                data_documents=docs,
                persist_dir=VECTOR_DIR,
                llm_name=LLM_NAME,
                embedding_name=EMB_NAME,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        st.success("Index prêt ✅")

with col2:
    question = st.text_input("💬 Pose ta question :", placeholder="Ex: Décrire le procédé de décantation primaire" , key="question")
    if question:
        with st.spinner("Génération de la réponse..."):
            index = build_or_load_index(
                data_documents=[],  # recharge l'index existant
                persist_dir=VECTOR_DIR,
                llm_name=LLM_NAME,
                embedding_name=EMB_NAME,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            answer, sources = ask_question(index, question, top_k=TOP_K, model_name=LLM_NAME)
        st.subheader("🧠 Réponse")
        st.write(answer)
        if sources:
            st.subheader("📚 Sources (extraits)")
            for s in dict.fromkeys(sources):  # unique in order
                st.code(str(s))

st.markdown("---")
st.caption("Tout tourne en local. Pense à lancer **Ollama** (\`ollama serve\`). Modèle recommandé : **mistral**.")
