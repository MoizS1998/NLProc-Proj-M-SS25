from transformers import pipeline

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes the Generator with a lightweight HuggingFace model.
        """
        self.model_name = model_name
        self.pipeline = pipeline("text2text-generation", model=model_name)
        self.prompt_template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    def build_prompt(self, context: str, question: str) -> str:
        """
        Formats the context and question into a prompt using the template.
        """
        return self.prompt_template.format(context=context.strip(), question=question.strip())

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generates an answer from the context and question using the model.
        """
        prompt = self.build_prompt(context, question)
        response = self.pipeline(prompt, max_length=256, truncation=True)
        return response[0]["generated_text"].strip()
