import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class GeminiLLM:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_answer(self, question, context):

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
reply:

"I couldn't find that information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.model.generate_content(prompt)

        return response.text