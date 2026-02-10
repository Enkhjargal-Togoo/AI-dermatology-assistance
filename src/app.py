# src/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_dermatology_pipeline

app = FastAPI(title="AI Dermatology Assistance")

class UserInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_symptom(data: UserInput):

    result = run_dermatology_pipeline(data.text)

    return result
