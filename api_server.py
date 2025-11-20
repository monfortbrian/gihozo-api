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

MODEL_ID = "epfl-llm/meditron-7b"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN")

COMMANDS = {
    "ACCESS": "[INST] You are a clinical assistant. Extract and summarize:\n\n{text} [/INST]",
    "ANALYZE": "[INST] You are an oncology specialist. Analyze:\n\n{text} [/INST]",
    "INTERPRET": "[INST] Provide structured clinical reasoning:\n\n{text} [/INST]",
    "REVIEW": "[INST] Review the case and list missing info:\n\n{text} [/INST]"
}


class ClinicalRequest(BaseModel):
    command: str
    patient_text: str


@app.get("/")
def root():
    return {"status": "Gihozo API running", "commands": list(COMMANDS.keys())}


@app.post("/process")
async def process(req: ClinicalRequest):
    if req.command not in COMMANDS:
        raise HTTPException(400, f"Invalid command: {req.command}")

    prompt = COMMANDS[req.command].format(text=req.patient_text)

    response = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7
            }
        },
        timeout=120
    )

    if response.status_code == 503:
        raise HTTPException(503, "Model loading... retry in 10-20 seconds")

    if response.status_code != 200:
        raise HTTPException(500, f"HuggingFace Router Error: {response.text}")

    data = response.json()

    # router returns array
    if isinstance(data, list) and "generated_text" in data[0]:
        output = data[0]["generated_text"]
    else:
        output = str(data)

    return {"command": req.command, "response": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
