import json
from pipeline import Pipeline  # adjust import if needed
import evaluate
import bert_score

def evaluate_pipeline(pipeline, eval_data_path: str):
    rouge = evaluate.load("rouge")
    bert_scores_list = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f.readlines()]
    predictions = []
    references = []
    for example in lines:
        question = example["question"]
        true_answer = example["answer"]
        pred = pipeline.ask(question, top_k=5)
        print(f"\nQ: {question}\n→ Prediction: {pred}\n→ Reference: {true_answer}\n")
        predictions.append(pred)
        references.append(true_answer)
    # ROUGE Score
    rouge_result = rouge.compute(predictions=predictions, references=references)
    print("🔍 ROUGE:", rouge_result)
    # BERTScore
    P, R, F1 = bert_score.score(predictions, references, lang="en")
    print("🔍 BERTScore:")
    print("  Precision:", round(P.mean().item(), 4))
    print("  Recall:   ", round(R.mean().item(), 4))
    print("  F1:       ", round(F1.mean().item(), 4))

if __name__ == "__main__":
    document_paths = ["documents/vikings.txt"]  # Adjust path!
    pipeline = Pipeline(document_paths=document_paths)  # this will call retriever.add_documents()
    evaluate_pipeline(pipeline, "eval_set.jsonl")