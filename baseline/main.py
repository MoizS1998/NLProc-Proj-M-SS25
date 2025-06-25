from retriever.retriever import Retriever
from generator.generator import Generator
import os

INDEX_PATH = "my_index"

def main():
    retriever = Retriever()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(base_dir,"retriever", "documents", "example.txt")

    if os.path.exists(INDEX_PATH + ".faiss"):
        retriever.load(INDEX_PATH)
        print("Loaded existing FAISS index.")
    else:
        if not os.path.isfile(doc_path):
            print(f"File not found: {doc_path}")
            return
        retriever.add_documents([doc_path])
        retriever.save(INDEX_PATH)
        print("Created and saved new FAISS index.")

    generator = Generator()

    question = input("Enter your question: ")
    context_chunks = retriever.query(question)
    context = "\n".join(context_chunks)

    prompt = generator.build_prompt(context, question)
    answer = generator.generate_answer(prompt)

    print("\n===== Prompt =====\n")
    print(prompt)
    print("\n===== Answer =====\n")
    print(answer)

if __name__ == "__main__":
    main()
