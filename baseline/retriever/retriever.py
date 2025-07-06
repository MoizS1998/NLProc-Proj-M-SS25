import os
import re
import pickle
import numpy as np
from typing import List
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", max_sentences=15, overlap=7):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.max_sentences = max_sentences
        self.overlap = overlap

    def _chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), self.max_sentences - self.overlap):
            chunk = " ".join(sentences[i:i + self.max_sentences])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

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

        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        dimension = normalized_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized_embeddings.astype('float32'))

    def query(self, question: str, top_k: int = 3) -> List[str]:
        query_embedding = self.model.encode([question])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k * 10)

        # Positional boosting: favor later chunks slightly
        total_docs = len(self.documents)
        candidate_chunks = []
        for idx, i in enumerate(indices[0]):
            score = similarities[0][idx]
            position_weight = 1.0 + (i / total_docs) * 0.1  # 10% bonus to last chunks
            boosted_score = score * position_weight
            candidate_chunks.append((self.documents[i], boosted_score))

        # Keyword-based soft reordering
        keywords = [w.lower() for w in question.split() if w.lower() not in ("what", "where", "who", "when", "why", "how") and len(w) > 2]
        keyword_chunks = [(chunk, score) for chunk, score in candidate_chunks if any(k in chunk.lower() for k in keywords)]
        non_keyword_chunks = [(chunk, score) for chunk, score in candidate_chunks if not any(k in chunk.lower() for k in keywords)]

        final_chunks = keyword_chunks + non_keyword_chunks

        # Fallback: always include last chunk if final chunks are weak
        if not final_chunks or final_chunks[0][1] < 0.25:
            final_chunks.append((self.documents[-1], 0.01))

        # Optional: debug output
        print("\n🔍 Final Retrieved Chunks:")
        for i, (chunk, score) in enumerate(final_chunks[:top_k]):
            print(f"\n[{i+1}] Score: {score:.4f}\n{chunk[:300]}{'...' if len(chunk) > 300 else ''}")

        return [chunk for chunk, _ in final_chunks[:top_k]]

    def save(self, path: str):
        faiss.write_index(self.index, path + ".faiss")
        with open(path + "_docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".faiss")
        with open(path + "_docs.pkl", "rb") as f:
            self.documents = pickle.load(f)
