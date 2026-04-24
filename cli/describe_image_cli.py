import argparse
import mimetypes
from urllib import response
from lib.search_utils import root_file
import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"



system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""


def main():
    parser = argparse.ArgumentParser(description="describe image CLI")
    parser.add_argument(
        "--query",
        type=str,
        help="Query string for describing the image",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="path to the image file to be described",
    )

    args = parser.parse_args()
    query = args.query
    image_path = args.image
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, 'rb') as f:
        image_data = f.read()
    
    parts = [
    system_prompt,
    types.Part.from_bytes(data=image_data, mime_type=mime),
    args.query.strip(),
]
    response = client.models.generate_content(
        model=model,
        contents=parts,
    )
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()