from __future__ import annotations
from typing import Dict, List, Text, Any
import json
import re
import requests

from rasa_sdk import Action, Tracker # type: ignore
from rasa_sdk.executor import CollectingDispatcher # type: ignore
from rasa_sdk.events import SlotSet, FollowupAction # type: ignore

from transformers import pipeline

# ──────────────────────────────────────────────────────────────────────────────
#  BioBERT pipelines (load once at import)
# ──────────────────────────────────────────────────────────────────────────────
SYMPTOM_CLF = pipeline(
    task="text-classification",
    model="fine-tuned-biobert-psoriasis-intent-v2",   # ← add hyphen
    tokenizer="dmis-lab/biobert-base-cased-v1.1",
)

NER = pipeline(
    task="token-classification",
    model="fine-tuned-biobert-ner-v2",                # ← add hyphen
    tokenizer="dmis-lab/biobert-base-cased-v1.1",
    aggregation_strategy="simple",
)
# ──────────────────────────────────────────────────────────────────────────────
#  Helper fns
# ──────────────────────────────────────────────────────────────────────────────
'''
IMAGE_CODE_RE = re.compile(r"image\s+reccode\s+(one|two)", re.I)

def classify_demo_image(text: Text) -> str | None:
    """Maps 'image reccode one/two' → diagnosis labels for the prototype."""
    m = IMAGE_CODE_RE.search(text)
    if not m:
        return None
    return "psoriasis" if m.group(1).lower() == "one" else "not psoriasis"
'''

def call_llama(prompt: Text) -> str:
    """Minimal wrapper around your Ollama/LLaMA HTTP endpoint."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "derm-assistant",
                "prompt": f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n",
                "stream": False,
                "options": {
                    "num_predict": 200,
                    "stop": ["<|im_end|>", "User:", "\nUser:", "\nAssistant:"],
                    "temperature": 0.4,
                    "top_p": 0.85
                    },
            },
            timeout=150,
        )
        resp.raise_for_status()
        return resp.json().get("response", "[backend empty]")
    except Exception as exc:  # noqa: BLE001
        return f"Backend error: {exc}"


class ActionPredictedIntentSymptoms(Action):

    def name(self):
        return "action_predicted_intent_symptoms"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_txt = tracker.latest_message.get('text')
        sender_id = tracker.sender_id

        symptoms = tracker.get_slot("symptom_text")
        image = tracker.get_slot("image_result")
        intent = tracker.latest_message.get("intent").get("name")
    
        if image and symptoms:
            return [FollowupAction("action_talk_to_llama")]
        else:
            # Step 1: Predict intent
            # BioBERT symptom classification
            clf_pred = SYMPTOM_CLF(user_txt)[0]  # [{'label': 'symptom', 'score': 0.93}]
            SYMPTOM_LABELS = {"symptom", "SYMPTOM", "describe_psoriasis"}  # ← tweak!
            if clf_pred["label"] in SYMPTOM_LABELS and clf_pred["score"] > 0.50:       
                return [
                    SlotSet("symptom_intent", "symptom_intent"),
                    FollowupAction("action_handle_user_input")
                    ]
            else:
                return [FollowupAction("action_talk_to_llama")]

# ──────────────────────────────────────────────────────────────────────────────
#  Action 1 – General small-talk / fallback to LLaMA
# ──────────────────────────────────────────────────────────────────────────────
class ActionTalkToLlama(Action):
    def name(self) -> Text:
        return "action_talk_to_llama"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
      
        user_message = tracker.latest_message.get('text')
        sender_id = tracker.sender_id

        interactions = []
        current_interaction = ""

        for event in tracker.events:
            if event.get("event") == "user":
                current_interaction += f"User: {event.get('text')}\n"
            elif event.get("event") == "bot":
                current_interaction += f"Bot: {event.get('text')}\n"
                interactions.append(current_interaction)
                current_interaction = ""

        # Last 3 interactions
        chat_history = "".join(interactions[-3:])
        print(f"[chat_history: {chat_history}")

        prompt = (
            f"A patient just asked the following question: '{user_message}'. "
            f"and the conversation so far:'{chat_history}'."
            f"As a helpful dermatology assistant, provide a short, clear, and friendly answer. "
            f"Keep it simple and easy to understand for a general audience."
            f"And stop the answer when you finish the principal idea, 150 words max. Summarize the idea."
        )
            
        dispatcher.utter_message(text=call_llama(prompt))
        return [FollowupAction("action_listen")]


# ──────────────────────────────────────────────────────────────────────────────
#  Action 2 – Handle ALL user input, run BioBERT where needed
# ──────────────────────────────────────────────────────────────────────────────
class ActionHandleUserInput(Action):
    def name(self) -> Text:
        return "action_handle_user_input"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        user_txt: Text = tracker.latest_message.get("text", "")
        events: List[Dict[Text, Any]] = []

        # Demo image code?
        #image_diag = classify_demo_image(user_txt)
        intent = tracker.latest_message.get("intent").get("name")
        intent2 = tracker.get_slot("symptom_intent")

        if intent == "image_diagnosis":
            dispatcher.utter_message(
                text=f"Thanks for the image –  **{user_txt}**."
            )
            events.append(SlotSet("image_result", user_txt))
            return events + [FollowupAction("action_check_completion")]

        # BioBERT symptom classification
        #clf_pred = SYMPTOM_CLF(user_txt)[0]  # [{'label': 'symptom', 'score': 0.93}]

        #SYMPTOM_LABELS = {"symptom", "SYMPTOM", "describe_psoriasis"}  # ← tweak!
        #if clf_pred["label"] in SYMPTOM_LABELS and clf_pred["score"] > 0.50:       

        elif intent == "describe_symptom" or intent2 == "symptom_intent":    
            dispatcher.utter_message(text="Got it, noted your symptoms.")
            # extract named entities for extra context
            ents = NER(user_txt)
            symptom_ents = [e["word"] for e in ents if e["entity_group"] == "SYMPTOM"]
            events += [
                SlotSet("symptom_text", user_txt),
                SlotSet("symptom_entities", json.dumps(symptom_ents)),
            ]
            return events + [
                FollowupAction("action_check_completion")
            ]
        # Not an image and not a symptom → small-talk fallback
        #dispatcher.utter_message(text=call_llama(user_txt))
        else:
            dispatcher.utter_message(response="utter_remember")
        return events


# ──────────────────────────────────────────────────────────────────────────────
#  Action 3 – Check if we have both pieces, steer the dialogue
# ──────────────────────────────────────────────────────────────────────────────
class ActionCheckCompletion(Action):
    def name(self) -> Text:
        return "action_check_completion"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        symptom = tracker.get_slot("symptom_text")
        image = tracker.get_slot("image_result")

        if symptom and image:
            dispatcher.utter_message(response="utter_complete_diagnosis")
            return [FollowupAction("action_process_symptom_photo")]

        if symptom and not image:
            dispatcher.utter_message(response="utter_ask_for_image")
            return [FollowupAction("action_listen")]
        elif image and not symptom:
            dispatcher.utter_message(response="utter_ask_for_symptoms")
            return [FollowupAction("action_listen")]

        return []


# ──────────────────────────────────────────────────────────────────────────────
#  Action 4 – Build final prompt, call LLaMA, reset slots
# ──────────────────────────────────────────────────────────────────────────────
class ActionProcessSymptomPhoto(Action):
    def name(self) -> Text:
        return "action_process_symptom_photo"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        symptom = tracker.get_slot("symptom_text")
        image = tracker.get_slot("image_result")
        ent_json = tracker.get_slot("symptom_entities") or "[]"
        entities = ", ".join(json.loads(ent_json))

        prompt = (
            "Act as a dermatologist.\n"
            f"Symptoms (verbatim): {symptom}\n"
            f"Structured entities: {entities or '–'}\n"
            f"Diagnosis based in Image analysis: {image}\n\n"
            "• Provide a concise preliminary assessment.\n"
            "• List 2-3 at-home care steps.\n"
            "• Say when to see a professional.\n"
            "150 words max."
        )
        dispatcher.utter_message(text=call_llama(prompt))
        
        return [
            SlotSet("symptom_text", symptom),
            SlotSet("symptom_entities", entities),
            SlotSet("image_result", image),
            SlotSet("symptom_intent",None)
        ]
        '''
            SlotSet("symptom_text", None),
            SlotSet("symptom_entities", None),
            SlotSet("image_result", None),
            SlotSet("symptom_intent",None)
        ]'''
