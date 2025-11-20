from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import time

app = FastAPI(
    title="Gihozo Doctor Co-Pilot API",
    description="Clinical reasoning for oncology specialists",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CONFIG - Using Meditron (works with HF Inference)
# Medical model with working API
HF_API_URL = "https://api-inference.huggingface.co/models/epfl-llm/meditron-7b"
HF_TOKEN = os.getenv("HF_TOKEN", "hf_hzKFLydMvlXjheoZBPklwnuDHeKKCxWn")

print("🩺 Gihozo API starting...")
print(f"Using model: epfl-llm/meditron-7b (medical reasoning)")
print(f"HF Token: Configured ✅")


# CLINICAL PROMPTS
COMMANDS = {
    "ACCESS": """<|im_start|>system
You are a clinical assistant helping an oncologist review patient cases.
<|im_end|>
<|im_start|>user
Extract and summarize key information from this patient case:

{text}

Provide a structured summary with:
- Demographics (age, sex)
- Chief complaint
- Key symptoms and timeline
- Medical history
- Next steps needed
<|im_end|>
<|im_start|>assistant
Summary:""",

    "ANALYZE": """<|im_start|>system
You are an oncology specialist assistant performing clinical reasoning.
<|im_end|>
<|im_start|>user
Analyze this patient case:

{text}

Provide:
1. Primary concerns
2. Differential diagnosis considerations
3. Staging clues (if cancer suspected)
4. Red flags requiring immediate attention
5. Clinical interpretation
<|im_end|>
<|im_start|>assistant
Analysis:""",

    "INTERPRET": """<|im_start|>system
You are a clinical decision support system for oncology.
<|im_end|>
<|im_start|>user
Provide structured clinical insights for this case:

{text}

Include:
- Clinical summary (2-3 sentences)
- Staging indicators (if applicable)
- Recommended investigations
- Risk factors present
- Missing information needed
<|im_end|>
<|im_start|>assistant
Interpretation:""",

    "REVIEW": """<|im_start|>system
You are a quality assurance assistant for oncology documentation.
<|im_end|>
<|im_start|>user
Review this case for completeness:

{text}

Identify what is missing:
- Staging parameters (TNM elements)
- Investigations (imaging, biopsy, markers)
- Clinical data (ECOG status, comorbidities)
- Treatment history details
<|im_end|>
<|im_start|>assistant
Review:"""
}


# REQUEST/RESPONSE MODELS
class ClinicalRequest(BaseModel):
    command: str
    patient_text: str


class HealthResponse(BaseModel):
    status: str
    model: str
    commands: list


# ENDPOINTS
@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "Gihozo API Running",
        "model": "epfl-llm/meditron-7b",
        "commands": list(COMMANDS.keys())
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api": "huggingface-inference",
        "model": "epfl-llm/meditron-7b"
    }


@app.post("/process")
async def process(request: ClinicalRequest):
    """
    Main endpoint for clinical reasoning
    Commands: ACCESS, ANALYZE, INTERPRET, REVIEW
    """

    # Validate command
    if request.command not in COMMANDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid command. Use: {list(COMMANDS.keys())}"
        )

    # Validate input
    if not request.patient_text.strip():
        raise HTTPException(
            status_code=400, detail="patient_text cannot be empty")

    # Build prompt
    prompt = COMMANDS[request.command].format(text=request.patient_text)

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    # Call HF API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True,
                        "return_full_text": False
                    }
                },
                timeout=90
            )

            # Handle model loading (503 error)
            if response.status_code == 503:
                error_data = response.json()
                if "loading" in str(error_data).lower():
                    wait_time = error_data.get("estimated_time", 20)
                    if attempt < max_retries - 1:
                        print(f"Model loading, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail=f"Model is loading. Please retry in {wait_time} seconds."
                        )

            # Handle successful response
            if response.status_code == 200:
                result = response.json()

                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get("generated_text", "")
                else:
                    generated = str(result)

                # Clean up response
                cleaned = generated.strip()

                return {
                    "command": request.command,
                    "response": cleaned if cleaned else generated
                }

            # Handle errors
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid HF token"
                )
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail="Model not found on HuggingFace"
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"HF API error: {response.text}"
                )

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(
                    f"⏳ Timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
                continue
            else:
                raise HTTPException(
                    status_code=504,
                    detail="Request timeout. Please try again."
                )

        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Network error: {str(e)}"
            )

    raise HTTPException(status_code=500, detail="Max retries exceeded")


# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
