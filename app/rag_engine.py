from typing import Optional, List
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

def ask_question(index: VectorStoreIndex, question: str, top_k: int = 4, model_name: str = "mistral"):
    """Interroge l'index avec un LLM local via Ollama et renvoie la réponse + sources."""
    llm = Ollama(model=model_name)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k, response_mode="compact")
    response = query_engine.query(question)

    # Try to extract sources (robust across versions)
    sources = []
    try:
        for sn in getattr(response, "source_nodes", []) or []:
            meta = sn.node.metadata or {}
            src = meta.get("file_path") or meta.get("filename") or meta.get("id") or "source"
            sources.append(src)
    except Exception:
        pass

    return str(response), sources
