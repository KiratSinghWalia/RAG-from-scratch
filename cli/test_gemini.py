import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def generate_content():
    prompt = "what is RAG?"
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt)
    
    print(response.text)
    print(f'Prompt tokens: {response.usage_metadata.prompt_token_count}')
    print(f'Response tokens: {response.usage_metadata.candidates_token_count}')


if __name__ == "__main__":
    generate_content()
    
