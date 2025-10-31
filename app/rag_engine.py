from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama


def ask_question(
    index: VectorStoreIndex,
    question: str,
    top_k: int = 4,
    model_name: str = "mistral",
    base_url: str = "http://127.0.0.1:11434",
    num_ctx: int = 2048,
    cpu_only: bool = False,
    max_tokens: int = 256,
    request_timeout_sec: int = 600,
):
    """Interroge l'index avec un LLM local via Ollama et renvoie la réponse + sources.

    Par défaut, on privilégie la compatibilité CPU (cpu_only=True) et un
    contexte réduit (num_ctx=2048) pour éviter les erreurs CUDA sur GPU avec
    faible VRAM.
    """

    additional_kwargs = {
        "num_ctx": num_ctx,
        "num_predict": max_tokens,
        # Force le modèle à répondre en français
        "system": "Tu es un assistant technique. Réponds toujours en français, de manière concise."
    }
    if cpu_only:
        additional_kwargs["num_gpu"] = 0

    llm = Ollama(
        model=model_name,
        base_url=base_url,
        additional_kwargs=additional_kwargs,
        request_timeout=request_timeout_sec,
    )
    query_engine = index.as_query_engine(
        llm=llm, similarity_top_k=top_k, response_mode="compact"
    )
    response = query_engine.query(question)

    # Extraction robuste des sources
    sources: List[str] = []
    try:
        for sn in getattr(response, "source_nodes", []) or []:
            meta = getattr(sn, "node", None)
            meta = getattr(meta, "metadata", {}) if meta is not None else {}
            src = meta.get("file_path") or meta.get("filename") or meta.get("id") or "source"
            sources.append(src)
    except Exception:
        pass

    return str(response), sources
