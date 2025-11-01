"""
Inspecteur de chunks indexés dans Chroma (persistance LlamaIndex).

Affiche les chunks (texte) avec leurs métadonnées principales (fichier d'origine,
identifiants, etc.). Permet un filtrage simple et une exportation CSV.

Exemples:
  python app/inspect_chunks.py --persist-dir vectorstore --limit 20
  python app/inspect_chunks.py --persist-dir vectorstore --group-by-file
  python app/inspect_chunks.py --persist-dir vectorstore --source-filter "test.txt"
  python app/inspect_chunks.py --persist-dir vectorstore --export-csv chunks.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

from chromadb import PersistentClient


def preview(text: str, max_len: int = 160) -> str:
    t = (text or "").replace("\n", " ").replace("\r", " ")
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def iter_collection_documents(collection, page_size: int = 200):
    """Itère sur tous les éléments d'une collection Chroma (paginé)."""
    offset = 0
    while True:
        # 'include' n'accepte pas 'ids' (toujours retourné implicitement)
        batch = collection.get(
            include=["metadatas", "documents"],
            limit=page_size,
            offset=offset,
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        metas = batch.get("metadatas") or []
        docs = batch.get("documents") or []
        for i, cid in enumerate(ids):
            yield cid, (metas[i] if i < len(metas) else {}), (docs[i] if i < len(docs) else "")
        offset += len(ids)


def main():
    ap = argparse.ArgumentParser(description="Inspecter les chunks indexés (Chroma)")
    ap.add_argument("--persist-dir", default="vectorstore", help="Dossier de persistance Chroma")
    ap.add_argument("--collection", default="eau_docs", help="Nom de la collection Chroma")
    ap.add_argument("--limit", type=int, default=50, help="Nombre max de chunks à afficher")
    ap.add_argument("--offset", type=int, default=0, help="Décalage de départ")
    ap.add_argument("--group-by-file", action="store_true", help="Grouper l'affichage par fichier source")
    ap.add_argument("--source-filter", default=None, help="Afficher uniquement les chunks dont la source contient ce texte")
    ap.add_argument("--export-csv", default=None, help="Chemin d'export CSV des chunks")
    args = ap.parse_args()

    if not os.path.isdir(args.persist_dir):
        raise SystemExit(f"Dossier de persistance introuvable: {args.persist_dir}")

    client = PersistentClient(path=args.persist_dir)
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        raise SystemExit(f"Collection introuvable: {args.collection}. Avez-vous indexé des documents ?")

    total = 0
    try:
        total = int(collection.count())
    except Exception:
        pass

    records: List[Dict[str, Any]] = []
    for cid, meta, doc in iter_collection_documents(collection):
        src = meta.get("file_path") or meta.get("filename") or meta.get("source") or meta.get("id") or ""
        if args.source_filter and (args.source_filter.lower() not in str(src).lower()):
            continue
        records.append({
            "id": cid,
            "source": src,
            "text": doc or "",
        })

    # Découpage offset/limit pour l'affichage
    view = records[args.offset : args.offset + args.limit] if args.limit is not None else records[args.offset :]

    print(f"Chunks trouvés: {len(records)} (total collection ~{total})")
    if not view:
        print("Aucun chunk à afficher (filtre trop restrictif ?)")
        return

    if args.group_by_file:
        # Grouper par source
        by_src: Dict[str, List[Dict[str, Any]]] = {}
        for r in view:
            by_src.setdefault(str(r["source"]), []).append(r)
        for src, rows in by_src.items():
            print(f"\n=== Source: {src} (chunks: {len(rows)}) ===")
            for r in rows:
                print(f"- {r['id']}: {preview(r['text'])}")
    else:
        # Liste plate
        for r in view:
            print(f"{r['id']} | {r['source']}\n  {preview(r['text'])}\n")

    # Export CSV optionnel
    if args.export_csv:
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "source", "text"])
            w.writeheader()
            for r in records:
                w.writerow(r)
        print(f"\nExport CSV: {args.export_csv}")


if __name__ == "__main__":
    main()
