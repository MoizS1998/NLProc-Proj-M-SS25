from flask import Flask, request, render_template
from pipeline import Pipeline
import os

app = Flask(__name__)

# Load all documents from the "documents" folder
DOCUMENT_DIR = "documents"
document_paths = [
    os.path.join(DOCUMENT_DIR, filename)
    for filename in os.listdir(DOCUMENT_DIR)
    if filename.endswith((".txt", ".md", ".pdf"))
]

# Initialize RAG pipeline with all documents
rag = Pipeline(document_paths=document_paths)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    question = ""
    
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            answer = rag.ask(question)
    
    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)