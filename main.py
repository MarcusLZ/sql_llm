from typing import Union
from fastapi import FastAPI
from model import *
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

model = None
tokenizer = None

app = FastAPI()
origins = ['*']

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    
    return {"Hello": "World"}

@app.get("/model_warmup")
def model_warmup():
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
        return {"message": "Model loaded"}
    
    return {"message": "Model already loaded"}

# Define request body model
class GenerateQueryRequest(BaseModel):
    question: str
    db_schema: str

@app.post("/generate_query")
def generate_query_api(request: GenerateQueryRequest):
    question = request.question
    db_schema = request.db_schema
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = prepare_prompt(question, db_schema)
    query = generate_query(prompt, model, tokenizer)
    
    return query

