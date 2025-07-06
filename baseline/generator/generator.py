from transformers import pipeline

class Generator:
    def __init__(self, model_name="google/flan-t5-large"):
        self.model_name = model_name
        self.pipeline = pipeline("text2text-generation", model=model_name)

        self.prompt_template = (
    "You are a thoughtful assistant. Read the context below carefully and answer the user's question in a detailed, insightful, and coherent paragraph. "
    "If the context does not provide enough information, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)



    def build_prompt(self, context: str, question: str) -> str:
        return self.prompt_template.format(
            context=context.strip(),
            question=question.strip()
        )




    def generate_answer(self, prompt: str) -> str:
        response = self.pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            truncation=False
        )
        return response[0]["generated_text"].strip()
