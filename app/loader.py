"""Utilitaires de chargement de documents.

Ce module fournit une fonction pour charger récursivement des documents
depuis un dossier à l'aide de `SimpleDirectoryReader` de LlamaIndex.
Il gère automatiquement plusieurs formats courants (PDF, DOCX, TXT, JSON, etc.).
"""

from typing import List, Optional, Sequence
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
import os


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

    # Initialise un lecteur récursif qui gère les formats courants et associe
    # l'ID du document au nom de fichier pour faciliter le suivi.
    reader_kwargs = dict(
        input_dir=data_path,
        recursive=True,
        filename_as_id=True,
    )
    if extensions:
        reader_kwargs["required_exts"] = list(extensions)
    try:
        reader_kwargs["num_workers"] = int(num_workers) if num_workers is not None else (os.cpu_count() or 4)
    except Exception:
        pass

    reader = SimpleDirectoryReader(**reader_kwargs)

    # Déclenche le chargement des fichiers en mémoire sous forme d'objets Document.
    docs = reader.load_data()

    # Agrège les documents dans la liste de sortie (permet des ajouts futurs).
    documents.extend(docs)

    # Retourne la collection complète des documents lus.
    return documents
