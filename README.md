import os
import time
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", embedding_dim)

# ---------------------------------------------------------
# Initialize Pinecone
# ---------------------------------------------------------
api_key = "pcsk_o5Yoa_J9E7WxpRA5h2nPeX95RzrperPCTvQqCTTgE8VhDpWD9mA51gGHavXRqpK6AsWY2"  # use env variable
pc = Pinecone(api_key=api_key)

index_name = "resume-search-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
print("Index ready")

# ---------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------
pdf_folder_path = r"./data_resume"   # ✅ your folder

def extract_text_from_pdfs(folder_path):
    documents = {}
    doc_id = 1

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            reader = PdfReader(file_path)

            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

            if text.strip():
                documents[f"doc_{doc_id}"] = text
                doc_id += 1

    return documents

start_time = time.time()
documents = extract_text_from_pdfs(pdf_folder_path)
end_time = time.time()

print(f"Text extraction completed in {end_time - start_time:.2f} seconds")
print(f"Total documents loaded: {len(documents)}")

# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------
doc_ids = list(documents.keys())
doc_texts = list(documents.values())

embeddings = model.encode(
    doc_texts,
    batch_size=16,
    show_progress_bar=True
).tolist()

vectors = list(zip(doc_ids, embeddings))

# ---------------------------------------------------------
# Upsert into Pinecone
# ---------------------------------------------------------
index.upsert(vectors=vectors)

# ---------------------------------------------------------
# Wait Until Indexing Completes
# ---------------------------------------------------------
def wait_until_indexing_complete(idx, expected_count, check_interval=5):
    while True:
        stats = idx.describe_index_stats()
        current_count = stats.total_vector_count
        print(f"Indexed: {current_count}/{expected_count}")
        if current_count >= expected_count:
            break
        time.sleep(check_interval)

wait_until_indexing_complete(index, len(documents))

# ---------------------------------------------------------
# Benchmark Queries
# ---------------------------------------------------------
queries = {
    "Sentence 1": "data engineering resume, azure data factory, azure databricks",
    "Sentence 2": "data science machine learning langchain gen ai agentic ai"
}

results_table = []

for label, query_text in queries.items():
    embed_start = time.time()
    query_embedding = model.encode(query_text).tolist()
    embed_end = time.time()

    search_start = time.time()
    index.query(vector=query_embedding, top_k=5, include_values=False)
    search_end = time.time()

    results_table.append({
        "Query": label,
        "Embedding Time": round(embed_end - embed_start, 4),
        "Pinecone Time": round(search_end - search_start, 4),
        "Total Time": round(search_end - embed_start, 4)
    })

print("\n==== PERFORMANCE ====\n")
for r in results_table:
    print(r)

# ---------------------------------------------------------
# Parallel Query Benchmark
# ---------------------------------------------------------
def pinecone_search(label, vector):
    start = time.time()
    index.query(vector=vector.tolist(), top_k=5)
    end = time.time()
    return {
        "Query": label,
        "Pinecone Time": round(end - start, 4)
    }

query_labels = list(queries.keys())
query_texts = list(queries.values())

embed_start = time.time()
embeddings = model.encode(query_texts, batch_size=16)
embed_end = time.time()

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(pinecone_search, label, emb)
        for label, emb in zip(query_labels, embeddings)
    ]

    for future in as_completed(futures):
        print(future.result())

print("\nDone ✅")
# Embedding
