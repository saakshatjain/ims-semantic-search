import requests

API_URL = "https://ims-semantic-search.onrender.com/retrieve"

resp = requests.post(
    API_URL,
    headers={"api-key": "ARS@btp"},
    json={"query": "innovision?", "prefetch_k": 20}
)

print(resp.json())
