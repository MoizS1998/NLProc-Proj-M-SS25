# 🧠 Multi-Hop Retrieval-Augmented Question Answering on Fanfiction

Welcome to **Team Musketeers'** final project for the Natural Language Processing course (SS25). This project tackles the challenge of answering complex questions from emotionally rich fanfiction and TV series texts using a Retrieval-Augmented Generation (RAG) system.

---

## 📌 Overview

This system takes a user question and generates an answer grounded in fan-authored documents (e.g., *Vikings*, *Harry Potter* fanfiction). The pipeline combines:

- Semantic chunk-based document retrieval
- Reranking with a cross-encoder
- Multi-hop context construction
- Answer generation using a pre-trained language model

---

## 📁 Project Structure

```
baseline/
├── app.py                   # Optional: Flask app for UI (requires Flask)
├── pipeline.py              # Main RAG pipeline
├── eval.py                  # Evaluation script
├── retriever/
│   └── retriever.py         # FAISS-based Retriever class
├── generator/
│   └── generator.py         # Generator using FLAN-T5
├── documents/               # Folder containing source .txt documents
│   └── *.txt
├── eval_set.jsonl           # Test set with QA pairs (ground truth)
├── logs/                    # Logs of past runs
```

---

## 🔧 Setup

### 1. Clone and Create Virtual Environment
```bash
git clone https://github.com/your-org/NLProc-Proj-M-SS25.git
cd NLProc-Proj-M-SS25/baseline
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> If you don’t have `requirements.txt`, manually install:
```bash
pip install numpy faiss-cpu sentence-transformers transformers nltk evaluate bert-score rouge-score flask
```

---

## 📜 Usage

### Run the Evaluation

Make sure you have `.txt` documents inside `documents/`, and that `eval_set.jsonl` exists in the `baseline/` folder.

```bash
python eval.py
```

It will automatically:
- Load the documents
- Index them
- Run the test questions
- Output the ROUGE and BERTScore metrics

---

## 🧠 Features

- ✅ **Improved Chunking**: Uses 25% overlap to preserve semantic continuity
- ✅ **Cross-Encoder Reranking**: Prioritizes relevant context
- ✅ **Multi-Hop Reasoning**: Merges multiple chunks (max 800 tokens) for complex answers
- ✅ **Grounded Prompting**: Guides the generator to stick to retrieved content

---

## 📊 Results

| Metric      | Value    |
|-------------|----------|
| ROUGE-1     | 0.1647   |
| ROUGE-2     | 0.0632   |
| ROUGE-L     | 0.1385   |
| ROUGE-Lsum  | 0.1374   |
| BERTScore-P | 0.8825   |
| BERTScore-R | 0.8381   |
| BERTScore-F1| 0.8595   |


## 📜 License

This project is for academic use only (Otto-Friedrich-Universität Bamberg, SS25).

---

## 📎 Acknowledgments

- Fanfiction content inspired by *Vikings* and *Harry Potter*.
- Inspired by RAG architectures like [REALM](https://arxiv.org/abs/2002.08909) and OpenAI's GPT pipeline integration.
