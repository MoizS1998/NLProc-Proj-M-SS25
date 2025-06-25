from transformers import pipeline

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes the Generator with a lightweight HuggingFace model.
        """
        self.model_name = model_name
        self.pipeline = pipeline("text2text-generation", model=model_name)

        # Custom fandom-style prompt
        self.prompt_template = (
            "You're a superfan of this series. Based on the plot below, answer in a fandom-friendly, emotional tone.\n\n"
            "Plot:\n{context}\n\n"
            "Fan Question: {question}\n"
            "Your Answer:"
        )

    def build_prompt(self, context: str, question: str) -> str:
        """
        Formats the context and question into a custom fandom prompt.
        """
        return self.prompt_template.format(
            context=context.strip(),
            question=question.strip()
        )

    def generate_answer(self, context: str, question: str) -> str:
        """
        Builds the prompt and generates an answer using the model.
        """
        prompt = self.build_prompt(context, question)
        response = self.pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            truncation=True
        )
        return response[0]["generated_text"].strip()
