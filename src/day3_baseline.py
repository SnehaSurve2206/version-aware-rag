from pathlib import Path
import time
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA = Path("data")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

chunks = pd.read_parquet(DATA / "chunks/chunks.parquet")
questions = pd.read_csv(RESULTS / "day2_benchmark_questions.csv")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = chunks["text"].fillna("").astype(str).tolist()
queries = questions["Question"].fillna("").astype(str).tolist()

print("Encoding corpus chunks...")
vectors = model.encode(
    texts, batch_size=16, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
).astype("float32")

print("Encoding questions...")
query_vectors = model.encode(
    queries, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
).astype("float32")

flat = faiss.IndexFlatIP(vectors.shape[1])
flat.add(vectors)

hnsw = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 80
hnsw.hnsw.efSearch = 64
hnsw.add(vectors)

rows = []
for qi, qrow in questions.iterrows():
    for method, index in [("Flat", flat), ("HNSW", hnsw)]:
        start = time.perf_counter()
        scores, ids = index.search(query_vectors[qi:qi+1], 10)
        latency = (time.perf_counter() - start) * 1000

        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
            chunk = chunks.iloc[int(idx)]
            rows.append({
                "question_id": qi + 1,
                "question_type": qrow["Type"],
                "question": qrow["Question"],
                "method": method,
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "category": chunk["category"],
                "document": chunk["document"],
                "version_raw": chunk["version_raw"],
                "chunk_text": chunk["text"],
                "query_latency_ms": latency
            })

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "day3_baseline_top10.csv", index=False)

print("\nBASELINE COMPLETE")
print("Questions:", len(questions))
print("Corpus chunks:", len(chunks))
print("Result rows:", len(out))
print("Saved: results/day3_baseline_top10.csv")
print(out.groupby("method")["query_latency_ms"].mean().round(3))
