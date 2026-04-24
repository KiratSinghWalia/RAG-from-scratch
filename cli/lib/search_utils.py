import json
from os import path
from pathlib import Path


root_file = Path(__file__).resolve().parents[2]
data_path = root_file/'data'/'movies.json'
stop_words_path =root_file/'data'/'stopwords.txt'
prompts_path = root_file/'cli'/'lib'/'prompts'

def load_movies():
    with open(data_path,"r") as f:
        data = json.load(f)
    return data['movies'] 


def load_stopwords():
    with open(stop_words_path,"r") as h:
        stopwatchdata = h.read().splitlines()
    return stopwatchdata



