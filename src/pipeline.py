# src/pipeline.py

def run_dermatology_pipeline(user_text: str, image_path: str | None = None):
    """
    Main multimodal pipeline (design only).

    Steps:
    1. Image analysis (if image provided)
    2. Symptom text understanding
    3. Medical retrieval (RAG)
    4. LLM response generation
    5. Safety filtering
    """

    result = {
        "image_features": None,
        "retrieved_docs": [],
        "llm_answer": "This is a placeholder response",
        "confidence": 0.0
    }

    return result
