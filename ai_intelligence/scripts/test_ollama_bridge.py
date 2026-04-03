import requests
import json

print("--- UTTSAV LOCAL AI GENERATIVE ENGINE ---")

# 1. Mock Data (This is what SHAP just gave us in the last test)
event_details = {
    "Risk_Level": "HIGH",
    "Event_Category": "Political Rally",
    "Expected_Crowd": 250000,
}

# The top 3 driving factors extracted from our SHAP Explainer
shap_reasons = [
    "Crowd Density is extremely high for the venue area.",
    "People Per Exit ratio exceeds safe limits.",
    "Capacity Utilization is over 100%."
]

# 2. Strict Prompt Engineering (To prevent LLM hallucinations)
prompt_template = f"""
You are an expert AI Safety Assistant for the Indian Government's UTTSAV Event Permission Portal.
An event application has been flagged with a {event_details['Risk_Level']} risk score.

Event Type: {event_details['Event_Category']}
Expected Crowd: {event_details['Expected_Crowd']}

The AI Risk Engine identified the following critical factors:
1. {shap_reasons[0]}
2. {shap_reasons[1]}
3. {shap_reasons[2]}

Task: Write a strict, professional 2-sentence recommendation for the government official reviewing this file. 
State the primary risk reason and suggest exactly one immediate precaution they should take before approving.
Do not use conversational filler. Be direct and authoritative.

Recommendation:
"""

# 3. Connect to Local Ollama API (Default port is 11434)
url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3",  # Change to "mistral" if you downloaded that instead
    "prompt": prompt_template,
    "stream": False
}

print("\nSending SHAP data to Local Ollama (llama3)...")
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("\n========================================================")
        print(">> OLLAMA GENERATED UI RECOMMENDATION:")
        print("========================================================")
        print(result['response'].strip())
        print("========================================================")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except requests.exceptions.ConnectionError:
    print("\n[!] Connection Error: Is Ollama currently running on your machine?")
    print("    Open your terminal and type: ollama run llama3")