# RAG Project – Summer Semester 2025

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## What it does:
- Takes a user question
- Retrieves relevant text chunks (simulated)
- Builds a structured prompt
- Uses a pre-trained LLM (`e.g: flan-t5-base`) to generate an answer
- Logs every step of the process
- Evaluates the result using test cases

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results
- `logs/`:  Stores logs of all RAG runs
- `utils/`: Helper functions shared across code

## Getting Started
1. Clone the repo
2. `cd baseline/`
3. Install dependencies: `pip install -r ../requirements.txt`

## Teams & Track
Group_id: Three Musketeers

## How to Run the Project
Run the following evaluation script:
python evaluation/evaluation.py


