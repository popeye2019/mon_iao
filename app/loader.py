import os
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

def load_documents(data_path: str) -> List[Document]:
    """Charge tous les documents lisibles depuis data_path (récursif).
    Supporte automatiquement PDF, DOCX, TXT, JSON, etc. via SimpleDirectoryReader.
    """
    documents: List[Document] = []
    # SimpleDirectoryReader gère la récursivité et la plupart des formats courants
    reader = SimpleDirectoryReader(input_dir=data_path, recursive=True, filename_as_id=True)
    docs = reader.load_data()
    documents.extend(docs)
    return documents
