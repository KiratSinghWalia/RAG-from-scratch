import os
import time
from dotenv import load_dotenv
from pathlib import Path
from lib.search_utils import prompts_path
import json
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)
model_name = "gemma-3-27b-it"

def Rerank(query,documents : dict):
    with open(prompts_path/'individual_rerank.md','r') as f:
        prompt = f.read()
    title = documents['title']
    description = documents['document']
    prompt = prompt.format(query=query,title=title,description=description)
    response = client.models.generate_content(
        model= model_name,
       contents=prompt)
    score = int(response.text.strip())
    time.sleep(3)  # Add a short delay to avoid hitting rate limits
    return score

def batch_rerank(query,documents : list):
    with open(prompts_path/'batch_rerank.md','r') as f:
        prompt = f.read()
    docs_str = ""
    for idx, doc in enumerate(documents):
        docs_str += f"{idx+1}. Movie_IDs: {doc['doc_id']}, Title: {doc['title']} ,Description: {doc['document']}."
    prompt = prompt.format(query=query,doc_list_str=docs_str)
    response = client.models.generate_content(
        model= model_name,
       contents=prompt)
    scores = response.text
    scores = json.loads(scores)
    return scores