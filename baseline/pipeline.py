import os
import json
import datetime
from typing import List
from retriever.retriever import Retriever
from generator.generator import Generator
from sentence_transformers import CrossEncoder

class Pipeline:
    def __init__(self, 
                 retriever_model: str = "all-MiniLM-L6-v2",
                 generator_model: str = "google/flan-t5-base",
                 document_paths: List[str] = None):
        
        self.retriever = Retriever(model_name=retriever_model)
        self.generator = Generator(model_name=generator_model)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        if document_paths:
            self.retriever.add_documents(document_paths)

    def ask(self, question: str, top_k: int = 5) -> str:
        # Step 1: Retrieve top-k chunks using FAISS
        context_chunks = self.retriever.query(question, top_k=top_k)

        # Step 2: Rerank chunks with cross-encoder
        scores = self.reranker.predict([(question, chunk) for chunk in context_chunks])
        reranked = [chunk for chunk, _ in sorted(zip(context_chunks, scores), key=lambda x: x[1], reverse=True)]

        # Step 3: Use top-1 chunk for answer generation
        context = reranked[0]
        prompt = self.generator.build_prompt(context, question)
        answer = self.generator.generate_answer(prompt)

        # Step 4: Log the interaction
        self._log({
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "retrieved_chunks": context_chunks,
            "reranked_chunk": context,
            "prompt": prompt,
            "generated_answer": answer,
            "group_id": "Team Musketeers"
        })

        return answer

    def add_documents(self, document_paths: List[str]):
        """Add new documents to the retriever"""
        self.retriever.add_documents(document_paths)

    def _log(self, data):
        os.makedirs("logs", exist_ok=True)
        with open("logs/run_log.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")

# Example usage
if __name__ == "__main__":
    rag = Pipeline(
        document_paths=["documents/example.txt"]
    )
    
    question = "Why did Daenerys burn King's Landing?"
    answer = rag.ask(question)
    
    print(f"Q: {question}")
    print(f"A: {answer}")