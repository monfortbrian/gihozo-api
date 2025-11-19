from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# CONFIG
MODEL_PATH = "monfortbrian/biomistral-7b-4bit-gihozo"  # or local ./biomistral-7b
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# APP INIT
app = FastAPI(title="Gihozo Doctor Co-Pilot API")


# MODEL LOADING (FP16 / CPU-safe)
print(f"Loading BioMistral model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)
model.to(DEVICE)
model.eval()
print("Model loaded!")


# INPUT SCHEMA
class PatientCommand(BaseModel):
    patient_text: str
    command: str  # ANALYZE, SUMMARIZE, INTERPRET, etc.


# API ENDPOINT
@app.post("/run-command")
async def run_command(data: PatientCommand):
    try:
        prompt = f"Patient case: {data.patient_text}\nCOMMAND:: {data.command}\nOUTPUT::"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
