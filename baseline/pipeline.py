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
                 generator_model: str = "google/flan-t5-large",
                 document_paths: List[str] = None):
        
        self.retriever = Retriever(model_name=retriever_model)
        self.generator = Generator(model_name=generator_model)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.verbose = False  

        if document_paths:
            self.retriever.add_documents(document_paths)

    def ask(self, question: str, top_k: int = 5) -> str:
        
        context_chunks = self.retriever.query(question, top_k=top_k)

        
        scores = self.reranker.predict([(question, chunk) for chunk in context_chunks])
        reranked = [chunk for chunk, _ in sorted(zip(context_chunks, scores), key=lambda x: x[1], reverse=True)]

    
        context = self._build_context(question, reranked, max_tokens=480)

# if all(score < 0.5 for score in scores):  {
#     context = "\n".join(context_chunks[:5])}

        prompt = self.generator.build_prompt(context, question)
        answer = self.generator.generate_answer(prompt)

    
        self._log({
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "retrieved_chunks": context_chunks,
            "reranked_chunk": context,
            "prompt": prompt,
            "generated_answer": answer,
            "group_id": "Team Musketeers"
        })
        
    
        if self.verbose:
            self._live_display(question, context_chunks, scores, reranked, prompt, answer)
            
        return answer




    def _build_context(self, question, reranked_chunks, max_tokens=480):
        tokenizer = self.generator.pipeline.tokenizer
        context = ""
        for chunk in reranked_chunks:
            new_context = context + "\n" + chunk
            input_ids = tokenizer.encode(f"Context:\n{new_context}\n\nQuestion: {question}", truncation=True)
            if len(input_ids) > max_tokens:
                break
            context = new_context  # This line should be inside the loop
        return context.strip()












    def add_documents(self, document_paths: List[str]):
        """Add new documents to the retriever"""
        self.retriever.add_documents(document_paths)

    def _log(self, data):
        os.makedirs("logs", exist_ok=True)
        with open("logs/run_log.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
            
    def _live_display(self, question, chunks, scores, reranked, prompt, answer):
        """Display pipeline results live in terminal"""
        print("\n" + "="*50)
        print(f"🔍 QUESTION: {question}")
        print("="*50)
        

        print("\n🔎 RETRIEVED CHUNKS (FAISS):")
        for i, chunk in enumerate(chunks):
            print(f"\nCHUNK {i+1}:\n{chunk[:150]}{'...' if len(chunk)>150 else ''}")
        
 
        print("\n⭐ RERANKED RESULTS (Cross-Encoder Scores):")
        for i, chunk in enumerate(reranked):
            print(f"\n#{i+1} [{scores[i]:.2f}]: {chunk[:150]}{'...' if len(chunk)>150 else ''}")
        

        print("\n🧠 GENERATOR INPUT:")
        print(f"TOP CHUNK: {reranked[0][:150]}{'...' if len(reranked[0])>150 else ''}")
        print(f"\nPROMPT:\n{prompt[:300]}{'...' if len(prompt)>300 else ''}")
        

        print("\n" + "="*50)
        print(f"💡 FINAL ANSWER: {answer}")
        print("="*50 + "\n")
    
    def set_verbose(self, verbose: bool = True):
        """Enable/disable live display mode"""
        self.verbose = verbose
