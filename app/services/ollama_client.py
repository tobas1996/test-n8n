import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
REQUEST_TIMEOUT = 120

def ollama_generate(payload: dict) -> dict:
    r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()
