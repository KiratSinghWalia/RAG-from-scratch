import os
from dotenv import load_dotenv
from pathlib import Path
from lib.search_utils import prompts_path


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def generate_content(prompt,query):
    prompt = prompt.format(query=query)
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt)
    return response.text 
    

def correct_spell(query):
    with open(prompts_path/'spelling.md','r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

def rewrite_query(query):
    with open(prompts_path/'rewrite.md','r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

#query expansion is one of the most common techniques that most of the companies misses. it is the process of expanding the original query with a
# additional terms or phrases that are semantically related to the original query. this can help to improve the recall of the search results by 
# including more relevant documents that may not have been retrieved with the original query alone.

def expand_query(query):
    with open(prompts_path/'expand.md','r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

def evaluate_results(query,documents):
    with open(prompts_path/'evaluate.md','r') as f:
        prompt = f.read()
    prompt = prompt.format(query=query,documents=documents)
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt,
    )
    return response.text

def RAG(query,documents,limit):
    with open(prompts_path/'rag.md','r') as f:
        prompt = f.read()
    prompt = prompt.format(query=query,docs=documents,limit=limit)
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt,
    )
    return response.text

def summarize(query,documents):
    with open(prompts_path/'summarize.md','r') as f:
        prompt = f.read()
    prompt = prompt.format(query=query,results=documents)
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt,
    )
    return response.text

def generate_citation(query,documents):
    with open(prompts_path/'citations.md','r') as f:
        prompt = f.read()
    prompt = prompt.format(query=query,documents=documents)
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt,
    )
    return response.text

def question_answering(query,documents):
    with open(prompts_path/'question.md','r') as f:
        prompt = f.read()
    prompt = prompt.format(question=query,context=documents)
    model_name = model
    response = client.models.generate_content(
        model=model_name,
       contents=prompt,
    )
    return response.text