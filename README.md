# IA locale — Traitement de l'eau (RAG)

Assistant documentaire **100% local** basé sur **Python + LlamaIndex + ChromaDB + Ollama + Streamlit**.

## 🚀 Démarrage rapide

1) **Installer Ollama** puis télécharger un modèle local (ex: `mistral`) :
```bash
# https://ollama.com/download
ollama pull mistral
ollama serve
```

2) **Créer et activer un environnement Python 3.11+**
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3) **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4) **Lancer l'interface Streamlit**
```bash
streamlit run app/main.py
```

5) **Indexation**
- Placez vos fichiers dans `./data` (PDF, DOCX, XLSX, JSON, TXT...).
- Cliquez sur le bouton **"Charger & indexer"** dans l'interface.
- Posez vos questions dans le champ dédié.

## 📁 Arborescence
```
mon_ia_eau/
 ├─ app/
 │   ├─ main.py               # Interface Streamlit
 │   ├─ loader.py             # Lecture des documents
 │   ├─ indexer.py            # Index LlamaIndex + Chroma
 │   ├─ rag_engine.py         # Moteur Q/R (RAG)
 │   └─ utils/config.py       # Chargement des paramètres
 ├─ data/                     # Vos fichiers techniques
 ├─ images/                   # Photos associées aux sites
 ├─ vectorstore/              # Stockage persistant Chroma
 ├─ settings.yaml             # Config projet
 ├─ requirements.txt
 └─ README.md
```

## 🧠 Notes
- Tout fonctionne **hors-ligne**.
- **LlamaIndex** gère le pipeline RAG (chargement, découpe, embeddings, retrieval, citations).
- **ChromaDB** stocke l'index vectoriel localement (persistant).
- **Ollama** exécute le LLM local (**mistral** recommandé).

## 🧩 Images et métadonnées
- Placez vos images sous `./images/<Site>/...`
- Référencez-les dans vos documents (nom de site, légendes) : elles seront proposées si le contexte le permet.

## 🐳 Docker (optionnel, plus tard)
- Vous pourrez dockeriser l'app avec le Dockerfile fourni dans `./docker`.
- Commencez sans Docker pour prototyper plus vite.
