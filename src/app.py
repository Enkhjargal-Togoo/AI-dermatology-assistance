# src/app.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Dermatology Assistance")

class UserInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_symptom(data: UserInput):
    """
    Placeholder endpoint for dermatology assistant.
    Later this will connect to:
    - LLM
    - RAG pipeline
    - Vision model output
    """
    return {
        "input": data.text,
        "message": "Dermatology AI assistance pipeline is under construction."
    }
