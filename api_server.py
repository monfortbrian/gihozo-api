from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import time
from typing import Optional

app = FastAPI(
    title="Gihozo Advanced Clinical Reasoning API",
    description="AI-powered clinical reasoning for oncology specialists",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],)


# CONFIG - HuggingFace Inference Endpoint
HF_ENDPOINT = "https://g68wtd4mixqxv1tn.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.getenv("HF_TOKEN", "")

print("Gihozo Advanced Clinical Reasoning API")
print(f"Endpoint: {HF_ENDPOINT}")
print(f"Token: {'Configured' if HF_TOKEN else 'MISSING'}")


# ADVANCED CLINICAL PROMPTS
COMMANDS = {
    "SUMMARIZE": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert clinical assistant helping oncologists review complex cancer cases. Your role is to extract and organize patient information systematically.<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT CASE:
{text}

TASK: Provide a comprehensive clinical summary organized as follows:

**DEMOGRAPHICS**
- Age, sex, relevant social history

**CHIEF COMPLAINT**
- Primary reason for presentation
- Duration and progression

**KEY CLINICAL FINDINGS**
- Physical examination findings
- Vital signs if mentioned
- Performance status (ECOG/Karnofsky if available)

**MEDICAL HISTORY**
- Prior cancers or treatments
- Comorbidities
- Medications
- Allergies

**FAMILY HISTORY**
- Cancer in family members
- Hereditary risk factors

**RISK FACTORS**
- Smoking, alcohol, occupational exposures
- Genetic predispositions

**CURRENT STATUS**
- Pending investigations
- Recent test results
- Current management

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**CLINICAL SUMMARY**

""",

    "REASONING": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a senior oncologist performing clinical reasoning on a complex case. Think systematically, identify patterns, and flag critical concerns.<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT CASE:
{text}

TASK: Perform deep clinical reasoning following this structure:

**PRIMARY CONCERNS**
What are the most worrying findings in this case? Why?

**DIFFERENTIAL DIAGNOSIS**
List possible diagnoses in order of likelihood. For each:
- Supporting evidence from the case
- Evidence against
- Key distinguishing features

**RED FLAGS**
Critical findings that require immediate attention or suggest poor prognosis:
- Clinical red flags (symptoms, exam findings)
- Tumor-related red flags (size, location, growth rate)
- Patient-related red flags (performance status, comorbidities)

**CONFLICTING OR UNCLEAR DATA**
Identify any inconsistencies or ambiguities in the case that need clarification.

**CLINICAL REASONING CHAIN**
Step-by-step thought process connecting symptoms → findings → likely diagnosis

**CERTAINTY LEVEL**
How confident are you in the primary diagnosis? What would increase certainty?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**CLINICAL REASONING ANALYSIS**

""",

    "STAGING": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an oncology specialist focused on cancer staging. Your expertise is in identifying staging indicators and gaps in staging workup.<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT CASE:
{text}

TASK: Provide detailed staging analysis:

**SUSPECTED CANCER TYPE**
Based on the case, what cancer is most likely?

**TNM STAGING INDICATORS**
Based on available information, analyze:

*T (Primary Tumor):*
- Size/extent clues from the case
- Local invasion indicators
- Missing T-stage information

*N (Regional Lymph Nodes):*
- Evidence of lymph node involvement
- Clinical vs pathological node status
- Missing N-stage information

*M (Distant Metastasis):*
- Any evidence of metastatic disease
- Sites that need evaluation
- Missing M-stage information

**PRELIMINARY STAGE ESTIMATE**
Based on available data, what stage is most likely? (I, II, III, IV)
Include confidence level and caveats.

**CRITICAL STAGING GAPS**
What information is ESSENTIAL but missing for accurate staging?

**PROGNOSTIC FACTORS**
Beyond TNM, what other factors affect prognosis in this case?
- Tumor grade/histology
- Molecular markers
- Performance status
- Age/comorbidities

**STAGING WORKUP NEEDED**
What tests are required to complete staging?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**ONCOLOGY STAGING ANALYSIS**

""",

    "PLAN": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a clinical decision support specialist helping oncologists determine the optimal diagnostic pathway.<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT CASE:
{text}

TASK: Develop a prioritized diagnostic plan:

**IMMEDIATE INVESTIGATIONS** (within 24-48 hours)
List urgent tests with justification:
- Test name
- Why it's urgent
- What it will reveal
- Expected findings

**SHORT-TERM INVESTIGATIONS** (within 1-2 weeks)
Essential but non-urgent tests:
- Test name
- Clinical rationale
- Expected findings
- Impact on management

**IMAGING RECOMMENDATIONS**
Specify modality, body regions, and contrast requirements:
- Which imaging? (CT, MRI, PET-CT, ultrasound, etc.)
- With or without contrast? Why?
- What are we looking for?

**LABORATORY TESTS**
- Tumor markers (which ones and why)
- Hematology/biochemistry needs
- Genetic/molecular testing if indicated

**PATHOLOGY/BIOPSY**
- Biopsy needed? What type? (core, FNA, excisional)
- Tissue analysis requirements
- Special stains or molecular testing

**URGENCY STRATIFICATION**
Rate overall case urgency: URGENT | SEMI-URGENT | ROUTINE
Justification for urgency level.

**DECISION TREE**
If test A is positive → next step is...
If test A is negative → consider...

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**DIAGNOSTIC PLAN**

""",

    "GUIDANCE": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a senior oncology consultant providing evidence-based management guidance. Consider best practices, guidelines (NCCN, ESMO, ASCO), and patient-specific factors.<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT CASE:
{text}

TASK: Provide comprehensive clinical guidance:

**INITIAL MANAGEMENT APPROACH**
What should be done first? Include:
- Immediate interventions (if any)
- Symptom management
- Patient counseling points

**MULTIDISCIPLINARY TEAM (MDT) DISCUSSION**
Should this case go to tumor board? What questions need answering?

**SPECIALIST REFERRALS NEEDED**
- Which specialists? (medical onc, surgical onc, radiation onc, palliative care, etc.)
- Urgency of referral
- Key questions for each specialist

**TREATMENT PATHWAY CONSIDERATIONS**
High-level treatment options (specific plans require full staging):
- Curative vs palliative intent
- Surgery candidacy
- Chemotherapy considerations
- Radiation therapy role
- Clinical trial eligibility

**RISK STRATIFICATION**
Classify patient risk: Low / Intermediate / High
Based on: tumor factors, patient factors, social factors

**FOLLOW-UP STRATEGY**
- How soon should patient be seen again?
- What should happen before next visit?
- Red flag symptoms to watch for

**SUPPORTIVE CARE NEEDS**
- Nutritional support
- Psychosocial support
- Palliative care involvement
- Social work/financial assistance

**PATIENT COMMUNICATION POINTS**
Key information to discuss with patient/family:
- What we know
- What we don't know yet
- Next steps
- Expected timeline
- Warning signs

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**CLINICAL GUIDANCE**

"""
}


# COMMAND METADATA
COMMAND_INFO = {
    "SUMMARIZE": {
        "title": "Clinical Summary",
        "description": "Extract and organize patient information systematically",
        "use_case": "New patient intake, case review, referral assessment",
        "output": "Structured summary with demographics, history, findings"
    },
    "REASONING": {
        "title": "Clinical Reasoning",
        "description": "Deep diagnostic analysis with red flag detection",
        "use_case": "Complex case analysis, differential diagnosis, critical thinking",
        "output": "Diagnostic hypotheses, rationale, red flags, reasoning chain"
    },
    "STAGING": {
        "title": "Oncology Staging",
        "description": "TNM staging analysis and prognostic assessment",
        "use_case": "Cancer staging, prognostic evaluation, staging workup planning",
        "output": "TNM indicators, stage estimate, staging gaps, prognostic factors"
    },
    "PLAN": {
        "title": "Diagnostic Plan",
        "description": "Prioritized investigation strategy with urgency levels",
        "use_case": "Diagnostic planning, test ordering, workup strategy",
        "output": "Imaging, labs, biopsies with urgency stratification"
    },
    "GUIDANCE": {
        "title": "Clinical Guidance",
        "description": "Evidence-based management recommendations",
        "use_case": "Treatment planning, MDT preparation, referral decisions",
        "output": "Management approach, referrals, follow-up, supportive care"
    }
}


# REQUEST/RESPONSE MODELS

class ClinicalRequest(BaseModel):
    command: Optional[str] = None
    patient_text: Optional[str] = None


class CommandInfo(BaseModel):
    command: str
    title: str
    description: str
    use_case: str
    output: str


class HealthResponse(BaseModel):
    status: str
    endpoint: str
    commands: dict


# HELPER FUNCTIONS

def validate_request(req: ClinicalRequest):
    """Validate and provide helpful error messages"""

    # Check if command is provided
    if not req.command:
        return {
            "error": "missing_command",
            "message": "No command specified.",
            "instructions": {
                "required": "Please provide a 'command' field in your request",
                "available_commands": list(COMMANDS.keys()),
                "example": {
                    "command": "REASONING",
                    "patient_text": "Your patient case here..."
                },
                "command_details": COMMAND_INFO
            }
        }

    # Check if command is valid
    if req.command not in COMMANDS:
        return {
            "error": "invalid_command",
            "message": f"'{req.command}' is not a valid command.",
            "instructions": {
                "you_provided": req.command,
                "available_commands": list(COMMANDS.keys()),
                "did_you_mean": get_closest_command(req.command),
                "command_details": COMMAND_INFO
            }
        }

    # Check if patient text is provided
    if not req.patient_text or not req.patient_text.strip():
        return {
            "error": "missing_patient_data",
            "message": "No patient information provided.",
            "instructions": {
                "required": "Please provide patient case information in the 'patient_text' field",
                "selected_command": req.command,
                "command_description": COMMAND_INFO[req.command]["description"],
                "example": {
                    "command": req.command,
                    "patient_text": "55-year-old female with left breast lump, 3cm, firm, irregular. Family history: mother had breast cancer at 60. Awaiting mammography and biopsy."
                },
                "tips": [
                    "Include patient demographics (age, sex)",
                    "Describe chief complaint and timeline",
                    "Mention relevant medical/family history",
                    "Include examination findings if available",
                    "Note any pending investigations"
                ]
            }
        }

    # All validations passed
    return None


def get_closest_command(input_cmd: str):
    """Suggest closest matching command"""
    input_lower = input_cmd.lower()
    suggestions = []

    mapping = {
        "summary": "SUMMARIZE",
        "summarize": "SUMMARIZE",
        "sum": "SUMMARIZE",
        "reason": "REASONING",
        "analyze": "REASONING",
        "analysis": "REASONING",
        "think": "REASONING",
        "stage": "STAGING",
        "staging": "STAGING",
        "tnm": "STAGING",
        "plan": "PLAN",
        "diagnostic": "PLAN",
        "workup": "PLAN",
        "tests": "PLAN",
        "guide": "GUIDANCE",
        "guidance": "GUIDANCE",
        "management": "GUIDANCE",
        "treatment": "GUIDANCE"
    }

    if input_lower in mapping:
        return mapping[input_lower]

    return None


# ENDPOINTS


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "🩺 Gihozo Advanced Clinical Reasoning API - LIVE",
        "endpoint": "HuggingFace Inference Endpoint (Dedicated GPU)",
        "commands": COMMAND_INFO
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "endpoint": HF_ENDPOINT,
        "token_configured": bool(HF_TOKEN),
        "commands_available": list(COMMANDS.keys())
    }


@app.get("/commands")
def list_commands():
    """Get detailed information about all commands"""
    return {
        "total_commands": len(COMMANDS),
        "commands": COMMAND_INFO
    }


@app.get("/commands/{command_name}")
def command_info(command_name: str):
    """Get information about a specific command"""
    command_upper = command_name.upper()
    if command_upper not in COMMAND_INFO:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Command not found",
                "available_commands": list(COMMANDS.keys())
            }
        )
    return COMMAND_INFO[command_upper]


@app.post("/process")
async def process(req: ClinicalRequest):
    """
    Main endpoint for advanced clinical reasoning

    Commands:
    - SUMMARIZE: Extract and organize patient information
    - REASONING: Deep diagnostic analysis with red flags
    - STAGING: TNM staging indicators and gaps
    - PLAN: Prioritized diagnostic investigations
    - GUIDANCE: Evidence-based management recommendations
    """

    # Validate request and provide helpful guidance
    validation_error = validate_request(req)
    if validation_error:
        return validation_error

    # Check HF token
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "configuration_error",
                "message": "API not properly configured. Contact administrator.",
                "details": "HF_TOKEN environment variable is missing"
            }
        )

    # Build prompt
    prompt = COMMANDS[req.command].format(text=req.patient_text.strip())

    # Prepare payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Call HF Endpoint with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(
                f"Processing {req.command} command (attempt {attempt + 1})...")

            response = requests.post(
                HF_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120
            )

            # Success
            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get("generated_text", str(result[0]))
                else:
                    generated = str(result)

                print(f"{req.command} completed ({len(generated)} chars)")

                return {
                    "command": req.command,
                    "command_info": COMMAND_INFO[req.command],
                    "response": generated.strip(),
                    "patient_text_length": len(req.patient_text),
                    "response_length": len(generated)
                }

            # Handle errors
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication failed. Invalid HF token."
                )

            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    print(f"Endpoint warming up...")
                    time.sleep(20)
                    continue
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Inference endpoint is warming up. Please retry in 20 seconds."
                    )

            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"HF Endpoint error: {response.text}"
                )

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            else:
                raise HTTPException(
                    status_code=504,
                    detail="Request timeout. Model might be processing a complex case."
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
    print(f"Starting Gihozo Advanced Clinical Reasoning API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
