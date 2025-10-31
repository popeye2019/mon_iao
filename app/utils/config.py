import os
import yaml

_DEFAULTS = {
    "paths": {
        "data_dir": "./data",
        "images_dir": "./images",
        "vectorstore_dir": "./vectorstore",
    },
    "model": {
        "llm_name": "mistral",
        "embedding_name": "nomic-embed-text",
    },
    "indexing": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 4,
    },
}

def load_config(path: str = "settings.yaml") -> dict:
    cfg = dict(_DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, file_cfg)

    # Allow env overrides
    cfg["model"]["llm_name"] = os.getenv("LLM_NAME", cfg["model"]["llm_name"])
    cfg["model"]["embedding_name"] = os.getenv("EMBEDDING_NAME", cfg["model"]["embedding_name"])
    cfg["paths"]["data_dir"] = os.getenv("DATA_DIR", cfg["paths"]["data_dir"])
    cfg["paths"]["images_dir"] = os.getenv("IMAGES_DIR", cfg["paths"]["images_dir"])
    cfg["paths"]["vectorstore_dir"] = os.getenv("VECTORSTORE_DIR", cfg["paths"]["vectorstore_dir"])
    return cfg

def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base
