from typing import Union
from fastapi import FastAPI
from model import *


model = None
tokenizer = None


app = FastAPI()

@app.get("/")
def read_root():
    
    return {"Hello": "World"}


@app.get("/model_warmup")
def model_warmup():
    global model, tokenizer
    model, tokenizer = load_model()
    
    return "Model loaded"


@app.post("/generate_query")
def generate_query(question: str, db_schema: str):
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = prepare_prompt(question, db_schema)
    query = generate_query(prompt, model, tokenizer)
    
    return query

