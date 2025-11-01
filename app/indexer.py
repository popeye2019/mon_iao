"""Construction et chargement d'index vectoriels.

Ce module configure LlamaIndex (LLM, embeddings, découpage en chunks)
et s'appuie sur Chroma pour la persistance. Il expose une fonction
`build_or_load_index` qui tente d'abord de charger un index existant
depuis le stockage persistant, puis le reconstruit à partir de documents
si nécessaire.
"""

import os
from typing import Optional, Sequence

from chromadb import PersistentClient
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


def build_or_load_index(
    data_documents: Optional[Sequence[Document]],
    persist_dir: str,
    llm_name: str = "mistral",
    embedding_name: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    ollama_base_url: str = "http://127.0.0.1:11434",
    llm_num_ctx: int = 2048,
    llm_num_gpu: Optional[int] = None,
    embedding_num_gpu: Optional[int] = None,
    request_timeout_sec: int = 600,
) -> VectorStoreIndex:
    """Charge un index persistant ou le construit depuis des documents.

    Paramètres:
    - data_documents: collection de `Document` à indexer (peut être vide/None si un index existe déjà).
    - persist_dir: dossier de persistance pour Chroma et les métadonnées d'index.
    - llm_name: nom du modèle LLM servi par Ollama pour les synthèses/questions.
    - embedding_name: nom du modèle d'embeddings servi par Ollama pour le vecteur.
    - chunk_size: taille des morceaux (tokens/caractères selon le splitter) pour le découpage.
    - chunk_overlap: recouvrement entre morceaux pour conserver le contexte local.

    Retourne:
    - Un `VectorStoreIndex` prêt à l'emploi, chargé depuis le stockage s'il existe,
      sinon reconstruit à partir de `data_documents` et persisté.
    """

    # Configuration globale de LlamaIndex: LLM, embeddings et stratégie de découpage.
    llm_kwargs = {"num_ctx": llm_num_ctx}
    if llm_num_gpu is not None:
        llm_kwargs["num_gpu"] = llm_num_gpu
    Settings.llm = Ollama(
        model=llm_name,
        base_url=ollama_base_url,
        additional_kwargs=llm_kwargs,
        request_timeout=request_timeout_sec,
    )

    embed_kwargs = {"keep_alive": "30m"}
    if embedding_num_gpu is not None:
        embed_kwargs["num_gpu"] = embedding_num_gpu
    Settings.embed_model = OllamaEmbedding(
        model_name=embedding_name,
        base_url=ollama_base_url,
        request_timeout=request_timeout_sec,
        ollama_additional_kwargs=embed_kwargs,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # S'assure que le dossier de persistance existe (idempotent).
    os.makedirs(persist_dir, exist_ok=True)

    # Initialise un client Chroma persistant pointant vers `persist_dir`.
    client = PersistentClient(path=persist_dir)

    # Récupère ou crée une collection Chroma nommée "eau_docs".
    try:
        collection = client.get_collection("eau_docs")
    except Exception:
        collection = client.create_collection("eau_docs")

    # Prépare le vector store pour LlamaIndex (contexte défini selon le scénario build/load).
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Si des documents sont fournis pendant l'étape d'indexation, on reconstruit directement
    # l'index puis on le persiste, sans tenter de charger un index inexistant.
    if data_documents:
        # IMPORTANT: pour la construction, ne pas fixer persist_dir dans StorageContext,
        # afin d'éviter toute tentative de lecture de docstore.json inexistant.
        storage_context_build = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        index = VectorStoreIndex.from_documents(
            data_documents,
            storage_context=storage_context_build,
        )
        index.storage_context.persist(persist_dir)
        return index

    # Sinon, on tente d'abord de CHARGER un index existant depuis le stockage persistant.
    try:
        storage_context_load = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir,
        )
        index = load_index_from_storage(storage_context=storage_context_load)
        return index
    except Exception:
        # Si l'index store n'est pas présent mais que des vecteurs existent déjà,
        # on reconstitue un index à partir du vector store (Chroma) uniquement.
        try:
            if collection.count() > 0:
                return VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context_load,
                )
        except Exception:
            pass

        # Sinon, on remonte une erreur explicite pour guider l'utilisateur.
        raise ValueError(
            "Aucun index existant détecté et aucun document fourni pour en créer un. "
            "Ajoute des fichiers dans 'data/' puis clique sur 'Charger & indexer'."
        )

def get_vector_count(persist_dir: str, collection_name: str = "eau_docs") -> int:
    """Retourne le nombre de vecteurs présents dans la collection Chroma.

    Si la collection ou le dossier n'existe pas, retourne 0.
    """
    try:
        client = PersistentClient(path=persist_dir)
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return 0
        try:
            return int(collection.count())
        except Exception:
            return 0
    except Exception:
        return 0
