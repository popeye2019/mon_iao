import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

from app.loader import load_documents
from app.indexer import build_or_load_index, get_vector_count
from llama_index.core.schema import Document
import yaml


def export_chunks(documents, export_path: Path, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except Exception:
        # If splitter not available, export raw documents
        splitter = None
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8", newline="") as f:
        if 'splitter' in locals() and splitter is not None:
            sp = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = sp.get_nodes_from_documents(documents)
            for n in nodes:
                rec = {
                    "id": getattr(n, "id_", None),
                    "text": getattr(n, "text", ""),
                    "metadata": getattr(n, "metadata", {}) or {},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            # Fallback: export document-level records (one line per Document)
            for d in documents:
                rec = {
                    "id": getattr(d, "doc_id", None),
                    "text": getattr(d, "text", ""),
                    "metadata": getattr(d, "metadata", {}) or {},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _fmt_val(v):
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ", ".join(_fmt_val(x) for x in v)
    if isinstance(v, dict):
        parts = []
        for sk, sv in v.items():
            parts.append(f"{sk} : {_fmt_val(sv)}")
        return "; ".join(parts)
    return str(v)


def _kv_text_and_meta(obj):
    meta = {}
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            val = _fmt_val(v)
            lines.append(f"{k} : {val}")
            meta[str(k)] = val
    elif isinstance(obj, list):
        val = ", ".join(_fmt_val(x) for x in obj)
        lines.append(val)
    else:
        lines.append(_fmt_val(obj))
    text = "\n".join(l for l in lines if l and str(l).strip())
    return text, meta


def _json_docs_from_file(path: Path) -> list[Document]:
    docs: list[Document] = []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return docs

    def emit(path_str: str, obj):
        text, meta = _kv_text_and_meta(obj)
        if not text:
            return
        header = f"[Source: {path.name} | JSON path: {path_str}]"
        final_text = f"{header}\n{text}"
        docs.append(
            Document(
                text=final_text,
                metadata={"file_path": str(path), "json_path": path_str, **meta},
            )
        )

    try:
        if isinstance(data, list):
            for i, item in enumerate(data):
                emit(f"$[{i}]", item)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        emit(f"$.{k}[{i}]", item)
                else:
                    emit(f"$.{k}", v)
        else:
            emit("$", data)
    except Exception:
        pass
    return docs


def _load_schema_mappings(schema_path: Path) -> list[dict]:
    try:
        data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    mappings: list[dict] = []
    try:
        sch = data.get("schemas", {}) or {}
        # Utiliser generic_record + autres si besoin
        for name in ("generic_record",):
            if name in sch:
                for m in sch[name].get("mappings", []) or []:
                    mappings.append(m)
    except Exception:
        return mappings
    return mappings


def _apply_mappings_to_obj(obj: dict, mappings: list[dict]) -> tuple[dict, list[str]]:
    """Retourne (meta_canon, canon_pairs) pour l'objet JSON.
    meta_canon: dict de paires canonisées (prefixées canon_)
    canon_pairs: liste de "Label : valeur" pour affichage compact
    """
    meta_canon: dict = {}
    pairs: list[str] = []
    if not isinstance(obj, dict):
        return meta_canon, pairs
    for m in mappings:
        src = m.get("source")
        target = m.get("target")
        if not src or not target:
            continue
        label = m.get("label", target)
        is_regex = bool(m.get("regex"))
        value = None
        try:
            if is_regex:
                # Cherche la première clé qui matche
                for k, v in obj.items():
                    if re.search(src, str(k)):
                        value = v
                        break
            else:
                if src in obj:
                    value = obj[src]
        except Exception:
            value = None
        if value is not None:
            val_s = _fmt_val(value)
            meta_canon[f"canon_{target}"] = val_s
            pairs.append(f"{label} : {val_s}")
    return meta_canon, pairs


def main():
    ap = argparse.ArgumentParser(description="Build index and optionally export chunks")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--persist-dir", required=True)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--embedding-model", required=True)
    ap.add_argument("--llm-num-ctx", type=int, default=2048)
    ap.add_argument("--embedding-num-gpu", default="None")
    ap.add_argument("--export-chunks", default=None, help="Path to write chunks (JSONL)")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Reduce noisy logs
    for name in [
        "httpx",
        "ollama",
        "chromadb",
        "llama_index",
        "llama_index.core",
        "llama_index.embeddings",
        "urllib3",
    ]:
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass

    # Build documents: JSON files -> one chunk per JSON record (enrichi);
    # autres fichiers via loader
    documents = []
    # Non-JSON via loader
    loaded = load_documents(str(data_dir))
    for d in loaded:
        fp = (d.metadata or {}).get("file_path") or ""
        if not str(fp).lower().endswith(".json"):
            documents.append(d)
    # Charger les mappings schema pour enrichir
    schema_path = Path(__file__).resolve().parents[1] / "ontology" / "schemas.yaml"
    mappings = _load_schema_mappings(schema_path) if schema_path.exists() else []

    # JSON files per record (enrichissement canon_ + ligne labels)
    for p in Path(data_dir).rglob("*.json"):
        raw_docs = _json_docs_from_file(p)
        if not mappings:
            documents.extend(raw_docs)
            continue
        # Enrichir chaque doc
        for d in raw_docs:
            obj_meta = dict(d.metadata or {})
            # Reconstituer un dict source minimal à partir des meta non canon_
            source_obj = {k: v for k, v in obj_meta.items() if k not in ("file_path", "json_path") and not str(k).startswith("canon_")}
            canon_meta, canon_pairs = _apply_mappings_to_obj(source_obj, mappings)
            if canon_meta:
                obj_meta.update(canon_meta)
            # Ajout d'une ligne compacte de labels canoniques (sans muter le Document d'origine)
            new_text = d.text or ""
            if canon_pairs:
                head = f"canon line : {' | '.join(canon_pairs)}"
                obj_meta["canon_line"] = " | ".join(canon_pairs)
                new_text = head + "\n" + new_text
            # Recréer un Document immuable avec le texte/metadata enrichis
            new_kwargs = {"text": new_text, "metadata": obj_meta}
            try:
                _id = getattr(d, "id_", None) or getattr(d, "doc_id", None)
                if _id:
                    new_kwargs["id_"] = _id
            except Exception:
                pass
            documents.append(Document(**new_kwargs))
    total = len(documents)
    if total == 0:
        print("Aucun document a indexer.")
        return 0

    # Optional export of chunks before inserting
    if args.export_chunks:
        export_chunks(
            documents,
            Path(args.export_chunks),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    processed = 0
    batch_size = max(1, min(64, math.ceil(total / 10)))
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        emb_gpu = None if str(args.embedding_num_gpu).lower() == "none" else int(args.embedding_num_gpu)
        _ = build_or_load_index(
            data_documents=batch,
            persist_dir=str(persist_dir),
            llm_name=str(args.llm_model),
            embedding_name=str(args.embedding_model),
            llm_num_ctx=int(args.llm_num_ctx),
            embedding_num_gpu=emb_gpu,
        )
        processed += len(batch)
        pct = int(processed * 100 / total)
        print(f"[{pct:3d}%] Indexation {processed}/{total}")

    count = get_vector_count(str(persist_dir))
    print(f"OK - vecteurs: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
