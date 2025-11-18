import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://ims-semantic-search.onrender.com/retrieve"
API_SECRET = os.getenv("API_SECRET")

resp = requests.post(
    API_URL,
    headers={"api-key": API_SECRET},
    json={"query": "responsible ai?", "prefetch_k": 50}
)

print(resp.json())
