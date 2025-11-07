from fastapi import FastAPI
from inference import ask_high_accuracy

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "service": "RAG Answer API"}

@app.get("/ask")
def ask(q: str):
    result = ask_high_accuracy(q)
    return {"query": q, "answer": result}
