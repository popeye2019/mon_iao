"""Utilitaires de chargement de documents.

Ce module fournit une fonction pour charger récursivement des documents
depuis un dossier à l'aide de `SimpleDirectoryReader` de LlamaIndex.
Il gère automatiquement plusieurs formats courants (PDF, DOCX, TXT, JSON, etc.).
"""

from typing import List, Optional, Sequence
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
import os
import json
from pathlib import Path
from app.text_normalize import expand_abbreviations


def load_documents(
    data_path: str,
    extensions: Optional[Sequence[str]] = None,
    num_workers: Optional[int] = None,
) -> List[Document]:
    """Charge tous les documents lisibles depuis un dossier (récursif).

    Paramètres:
    - data_path: chemin du dossier racine contenant les fichiers à indexer.

    Retourne:
    - Une liste de `Document` (objets LlamaIndex) résultant de la lecture des
      fichiers trouvés dans `data_path` et ses sous-dossiers.

    Remarques:
    - `SimpleDirectoryReader` détecte automatiquement de nombreux formats
      (PDF, DOCX, TXT, JSON, etc.) et parcourt récursivement l'arborescence.
    - Les identifiants de documents sont dérivés des noms de fichiers via
      `filename_as_id=True` pour une traçabilité simple.
    """

    # Contiendra tous les documents chargés depuis le dossier cible.
    documents: List[Document] = []

    root = Path(data_path)
    if not root.exists():
        return documents

    # Partitionne les fichiers en JSON et non-JSON
    all_files = [p for p in root.rglob('*') if p.is_file()]
    json_files = [p for p in all_files if p.suffix.lower() == '.json']
    other_files = [p for p in all_files if p.suffix.lower() != '.json']

    # 1) Charger les fichiers non-JSON via SimpleDirectoryReader en ciblant explicitement les fichiers
    if other_files:
        input_list = [str(p) for p in other_files]
        if extensions:
            allowed = set(e.lower() for e in extensions)
            input_list = [f for f in input_list if Path(f).suffix.lower() in allowed]
        if input_list:
            reader_kwargs = dict(
                input_files=input_list,
                filename_as_id=True,
            )
            if num_workers is not None:
                try:
                    reader = SimpleDirectoryReader(**{**reader_kwargs, 'num_workers': int(num_workers)})
                except TypeError:
                    reader = SimpleDirectoryReader(**reader_kwargs)
            else:
                reader = SimpleDirectoryReader(**reader_kwargs)
            for d in reader.load_data():
                documents.append(
                    Document(text=expand_abbreviations(d.text), metadata=d.metadata)
                )

    # 2) Charger les JSON avec une logique par enregistrement/section et json_path
    def iter_string_leaves(obj):
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, (int, float)):
            yield str(obj)
        elif isinstance(obj, list):
            for it in obj:
                yield from iter_string_leaves(it)
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from iter_string_leaves(v)

    def make_kv_text_and_meta(obj) -> (str, dict):
        """Rend l'objet sous forme lisible "cle: valeur" et un dict metadata plat.

        - Si obj est un dict: une ligne par cle primaire; valeurs scalaires/strings
          sont rendues directement, les listes sont jointes par ", ", les dicts
          imbriqués sont aplanis en paires "souscle: valeur".
        - Si obj est une liste: concatène les éléments (applique les mêmes règles).
        - Sinon: retourne str(obj).
        """
        meta = {}

        def fmt_val(v):
            if v is None:
                return ""
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                return ", ".join(fmt_val(x) for x in v)
            if isinstance(v, dict):
                # aplatir une profondeur
                parts = []
                for sk, sv in v.items():
                    parts.append(f"{sk}: {fmt_val(sv)}")
                return "; ".join(parts)
            return str(v)

        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                val = fmt_val(v)
                lines.append(f"{k}: {val}")
                # Dupliquer la paire dans metadata (stringifiée)
                try:
                    meta[str(k)] = val
                except Exception:
                    pass
        elif isinstance(obj, list):
            # Ligne unique avec elements
            val = ", ".join(fmt_val(x) for x in obj)
            lines.append(val)
        else:
            lines.append(fmt_val(obj))

        return "\n".join(l for l in lines if l and str(l).strip()), meta

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        def emit_record(path_str: str, obj):
            text, meta_extra = make_kv_text_and_meta(obj)
            if not text:
                return
            documents.append(
                Document(
                    text=expand_abbreviations(text),
                    metadata={
                        'file_path': str(jf),
                        'json_path': path_str,
                        **meta_extra,
                    },
                )
            )

        try:
            if isinstance(data, list):
                for i, item in enumerate(data):
                    emit_record(f'$[{i}]', item)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            emit_record(f'$.{k}[{i}]', item)
                    else:
                        emit_record(f'$.{k}', v)
            else:
                emit_record('$', data)
        except Exception:
            pass

    return documents
