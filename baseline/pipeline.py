from typing import List
from retriever.retriever import Retriever
from generator.generator import Generator  # Your Generator class

class Pipeline:
    def __init__(self, 
                 retriever_model: str = "all-MiniLM-L6-v2",
                 generator_model: str = "google/flan-t5-base",
                 document_paths: List[str] = None):
        
        self.retriever = Retriever(model_name=retriever_model)
        self.generator = Generator(model_name=generator_model)
        
        if document_paths:
            self.retriever.add_documents(document_paths)

    def ask(self, question: str, top_k: int = 3) -> str:
       
        # Retrieve relevant information
        context_chunks = self.retriever.query(question, top_k=top_k)
        context = "\n".join(context_chunks)
        
        # Generate answer
        prompt = self.generator.build_prompt(context, question)
        return self.generator.generate_answer(prompt)

    def add_documents(self, document_paths: List[str]):
        """Add new documents to the retriever"""
        self.retriever.add_documents(document_paths)

# Example usage
if __name__ == "__main__":
   
    rag = Pipeline(
        document_paths=["documents/example.txt"]
    )
    
    # Ask a question
    question = "What is the capital of Italy?"
    answer = rag.ask(question)
    
    print(f"Q: {question}")
    print(f"A: {answer}")
