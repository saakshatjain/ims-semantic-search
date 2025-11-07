import requests

API_URL = "https://ims-semantic-search.onrender.com/retrieve"

resp = requests.post(
    API_URL,
    headers={"api-key": "api_key"},
    json={"query": "innovision?", "prefetch_k": 1}
)

print(resp.json())
