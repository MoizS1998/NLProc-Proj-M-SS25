import os
import re
import pickle
import numpy as np  # Added for vector normalization
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def _chunk_text(self, text: str):
        chunks = re.split(r'\n(?=\d+\.)', text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _load_file(self, filepath: str) -> str:
        if filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath.endswith(".pdf"):
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def add_documents(self, filepaths: List[str]):
        self.documents = []
        for filepath in filepaths:
            text = self._load_file(filepath)
            chunks = self._chunk_text(text)
            self.documents.extend(chunks)

        # Generate and normalize embeddings
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Create FAISS index with inner product (cosine similarity)
        dimension = normalized_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized_embeddings.astype('float32'))

    def query(self, question: str, top_k: int = 3) -> List[str]:
        # Normalize query embedding
        query_embedding = self.model.encode([question])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search with cosine similarity
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        return [self.documents[i] for i in indices[0]]

    def save(self, path: str):
        faiss.write_index(self.index, path + ".faiss")
        with open(path + "_docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".faiss")
        with open(path + "_docs.pkl", "rb") as f:
            self.documents = pickle.load(f)
