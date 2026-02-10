# AI Dermatology Assistant (Multimodal)
This project introduces the Skin Care Assistant, a multimodal, AI-powered web application designed to support the early detection of dermatological disorders. In its initial phase, the system focuses on psoriasis and combines computer vision with natural language processing (NLP) to deliver an accessible, interactive user experience.

## Problem
Help users describe skin symptoms and optionally analyze images to provide safe, educational guidance and triage suggestions.

## Solution
A multimodal pipeline combining:
- Image analysis (vision model)
- Medical text understanding (LLM)
- Safety guardrails and disclaimers

## Architecture
<img width="1326" height="753" alt="image" src="https://github.com/user-attachments/assets/f1f28374-9a5b-426d-8e9b-690ad4b6c7ca" />


## Tech Stack
Python, Fine-tuned Biobert model, Fine-tuned Llama model, Fine-tuned Yolov8n model, Rasa framework, FastAPI connections, NextJS

## How to Run
This project contains three main components:
- Rasa chatbot (rasa_app/)
- BioBERT fine-tuning and inference code (Biobert/)
- API / orchestration layer (src/)
  
### 1. Create environment
python -m venv .venv
source .venv/bin/activate
.venv\Scripts\activate
### 2. Intall Dependencies
pip install -r requirements.txt
### 3. Run rasa chatbot
cd rasa_app
rasa train
rasa run --enable-api
### 4. For more detailed instructions: 
Please see Chatbot Guideline.doc

## Results
<img width="975" height="543" alt="image" src="https://github.com/user-attachments/assets/aa023fca-8ce3-4c4c-a597-af9a6c02c902" />

## Limitations
Not a medical diagnosis tool.

## Contrubutions:
It is a group project including all other group members. So I uploaded only Rasa chatbot part and fine-tuned Biobert model part. 

## Model files
Trained BioBERT model checkpoints are not stored in this repository due to file size limits.
This repository contains training and evaluation code only.

## Next Steps
- Improve dataset
- Better evaluation
- Add monitoring & logging
- Changing the Rasa framework to Langchain/LangGraph
- Retrieval-Augmented Generation (RAG) from trusted sources
- Add more skin desease diagnosis
