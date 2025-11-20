# api_server.py - FINAL WORKING VERSION
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Gihozo Clinical Reasoning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Used Llama-2-7B
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.getenv("HF_TOKEN", "hf_hzKFLydMvlXjheoZBPklwnuDHeKKCxWn")

COMMANDS = {
    "ACCESS": "[INST] You are a clinical assistant. Extract and summarize this patient case:\n\n{text}\n\nProvide: Demographics, Chief complaint, Key symptoms, Medical history, Next steps [/INST]",
    "ANALYZE": "[INST] You are an oncology specialist. Analyze this case:\n\n{text}\n\nProvide: Primary concerns, Differential diagnosis, Staging clues, Red flags [/INST]",
    "INTERPRET": "[INST] You are a clinical decision support system. Provide structured insights:\n\n{text}\n\nInclude: Clinical summary, Staging indicators, Recommended investigations, Risk factors [/INST]",
    "REVIEW": "[INST] You are a quality assurance assistant. Review this case for missing information:\n\n{text}\n\nIdentify missing: Staging parameters, Investigations, Clinical data [/INST]"
}


class ClinicalRequest(BaseModel):
    command: str
    patient_text: str


@app.get("/")
def root():
    return {"status": "Gihozo API", "model": "Llama-2-7b-chat", "commands": list(COMMANDS.keys())}


@app.post("/process")
async def process(req: ClinicalRequest):
    if req.command not in COMMANDS:
        raise HTTPException(400, f"Invalid command: {req.command}")

    prompt = COMMANDS[req.command].format(text=req.patient_text)

    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": prompt, "parameters": {
            "max_new_tokens": 300, "temperature": 0.7}},
        timeout=60
    )

    if response.status_code == 503:
        raise HTTPException(503, "Model loading, retry in 20 seconds")
    elif response.status_code == 200:
        result = response.json()
        text = result[0]["generated_text"] if isinstance(
            result, list) else str(result)
        return {"command": req.command, "response": text}
    else:
        raise HTTPException(500, f"Error: {response.text}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
