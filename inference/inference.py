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
    sql = """
    select id, notice_id, chunk_text,
           1 - (embedding <=> %(vec)s) as similarity
    from notice_chunks
    order by embedding <=> %(vec)s
    limit %(top_k)s;
    """

    res = supabase.postgres.rpc("exec_sql", {
        "query": sql,
        "params": {"vec": query_vec, "top_k": top_k}
    }).execute()

    return res.data

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
    top_chunks = [r[0]["chunk_text"] for r in ranked[:4]]

    # Step 4: merge context
    context = "\n\n".join(top_chunks)

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
