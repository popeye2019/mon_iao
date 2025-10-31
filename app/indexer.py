# app/indexer.py
import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from chromadb import PersistentClient

def build_or_load_index(
    data_documents,
    persist_dir: str,
    llm_name: str = "mistral",
    embedding_name: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> VectorStoreIndex:
    # Config LlamaIndex
    Settings.llm = Ollama(model=llm_name)
    Settings.embed_model = OllamaEmbedding(model_name=embedding_name)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)

    # collection Chroma persistante
    try:
        collection = client.get_collection("eau_docs")
    except Exception:
        collection = client.create_collection("eau_docs")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Essaye de CHARGER d'abord
    try:
        index = load_index_from_storage(storage_context=storage_context)
        return index
    except Exception:
        # Si on a des documents, on (re)construit alors l'index
        if data_documents:
            index = VectorStoreIndex.from_documents(data_documents, storage_context=storage_context)
            index.storage_context.persist(persist_dir)
            return index
        # Sinon, message clair
        raise ValueError(
            "Aucun index existant détecté et aucun document fourni pour en créer un. "
            "Ajoute des fichiers dans 'data/' puis clique sur 'Charger & indexer'."
        )
