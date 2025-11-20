import requests

API_URL = "https://gihozo-api-production.up.railway.app/process"

patient_case = """
55-year-old female
Left breast lump, 3cm, firm, irregular
Family history: mother had breast cancer at 60
Pending mammography and biopsy
"""

response = requests.post(
    API_URL,
    json={
        "command": "ACCESS",
        "patient_text": patient_case
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
