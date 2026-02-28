import os
from supabase import create_client
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")

# Load your LoRA-tuned Qwen
tokenizer = AutoTokenizer.from_pretrained("./model/qwen-lora", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./model/qwen-lora", trust_remote_code=True)

def supabase_vector_query(query_vec, top_k=20):
    # Call the actual RPC we just fixed, which correctly queries notice_chunks_new_2
    res = supabase.rpc(
        "match_notice_chunks",
        {"query_embedding": query_vec, "match_count": top_k}
    ).execute()
    return res.data or []

def ask_high_accuracy(question: str):
    # Step 1: embed query
    qvec = embedder.encode(question).tolist()

    # Step 2: fetch top chunks
    rows = supabase_vector_query(qvec, top_k=20)

    if not rows:
        return "No relevant information found."

    # Step 3: rerank
    pairs = [[question, r["chunk_text"]] for r in rows]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
    
    top_chunks = []
    for r, score in ranked[:4]:
        title = r.get("notice_title") or "Unknown Document"
        text = r.get("chunk_text") or ""
        top_chunks.append(f"[Source: {title}]\n{text}")

    # Step 4: merge context
    context = "\n\n---\n\n".join(top_chunks)

    # Step 5: run through LLM
    prompt = f"""
You are an assistant answering based only on this context. Do not hallucinate!

### CONTEXT ###
{context}

### QUESTION ###
{question}

### ANSWER ###
"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=1024)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer.replace(prompt, "").strip()
