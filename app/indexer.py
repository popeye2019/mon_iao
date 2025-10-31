import os
from typing import Optional
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
    """Crée ou recharge un index LlamaIndex adossé à Chroma (persistant)."""

    # Configure global Settings for LlamaIndex
    Settings.llm = Ollama(model=llm_name)
    Settings.embed_model = OllamaEmbedding(model_name=embedding_name)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Chroma persistent store
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)

    # Get or create collection
    try:
        collection = client.get_collection("eau_docs")
    except Exception:
        collection = client.create_collection("eau_docs")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # If there is existing storage state, load the index;
    # otherwise create a new one from documents.
    storage_files = [f for f in os.listdir(persist_dir) if os.path.isfile(os.path.join(persist_dir, f))]
    if storage_files:
        index = load_index_from_storage(storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_documents(data_documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir)

    return index
