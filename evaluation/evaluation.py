import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime
import os
from baseline.generator.generator import Generator 


LOG_PATH = "logs/rag_log.json"
TEST_INPUT_PATH = "evaluation/test_inputs.json"

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# 1. --- LOGGING FUNCTION ---
def log_rag_run(question, retrieved_chunks, prompt, generated_answer, group_id):
    log_entry = {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": generated_answer,
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# 2. --- LOADING TEST QUESTIONS ---
def load_test_questions():
    with open(TEST_INPUT_PATH, "r") as f:
        return json.load(f)


# 3. --- TEST FUNCTION ---
def test_pipeline(generator, test_cases, group_id="Three_Musketeers"):
    for case in test_cases:
        question = case["question"]
        expected = case["expected_answer_contains"]
        
    
        retrieved_chunks = case["retrieved_chunks"]
        prompt = generator.build_prompt("\n".join(retrieved_chunks), question)
        answer = generator.generate_answer("\n".join(retrieved_chunks), question)

        # Logging the run
        log_rag_run(
            question=question,
            retrieved_chunks=retrieved_chunks,
            prompt=prompt,
            generated_answer=answer,
            group_id=group_id
        )

        # Testing output
        print(f"\nQ: {question}")
        print(f"Expected: {expected}")
        print(f"Answer: {answer}")
        print("✅ Pass" if expected.lower() in answer.lower() else "❌ Fail")


# 4. --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load your generator 
    generator = Generator(model_name="google/flan-t5-base")

    # Load test questions
    test_cases = load_test_questions()

    # Run logging and testing
    test_pipeline(generator, test_cases)
