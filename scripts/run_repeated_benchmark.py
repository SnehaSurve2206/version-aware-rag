"""Repeatable Flat-vs-HNSW benchmark for version-aware-rag.
Run from the project root with the project's .venv active:
    python scripts\\run_repeated_benchmark.py
Inputs expected:
    data\\chunks\\chunks.parquet OR data\\chunks\\chunks_export.csv
    results\\day3_baseline_top10.csv (for the 50 query texts)
Outputs:
    results\\day3_repeated_benchmark_summary.csv
    results\\day3_repeated_benchmark_per_query.csv
    results\\day3_repeated_benchmark_rankings.csv
    results\\day3_repeated_benchmark_index_sizes.csv
This measures repeated retrieval timing; it does not establish relevance labels.
"""
from pathlib import Path
import time, os, statistics, json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path.cwd()
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
parquet = ROOT / "data" / "chunks" / "chunks.parquet"
csv = ROOT / "data" / "chunks" / "chunks_export.csv"
base_path = RESULTS / "day3_baseline_top10.csv"
if not base_path.exists():
    raise SystemExit(f"Missing {base_path}. Run from project root.")
if parquet.exists():
    corpus = pd.read_parquet(parquet)
elif csv.exists():
    corpus = pd.read_csv(csv)
else:
    raise SystemExit("Could not find data/chunks/chunks.parquet or chunks_export.csv")
base = pd.read_csv(base_path)
required_corpus = {"chunk_id", "text", "category", "document", "version_raw"}
if not required_corpus.issubset(corpus.columns):
    raise SystemExit(f"Corpus missing columns: {sorted(required_corpus-set(corpus.columns))}")
required_base = {"question_id", "question", "method", "rank", "chunk_id"}
if not required_base.issubset(base.columns):
    raise SystemExit(f"Baseline missing columns: {sorted(required_base-set(base.columns))}")

# One query per question, stable order.
questions = base[["question_id", "question"]].drop_duplicates().sort_values("question_id").reset_index(drop=True)
texts = corpus["text"].fillna("").astype(str).tolist()
ids = corpus["chunk_id"].astype(str).tolist()
if len(set(ids)) != len(ids):
    raise SystemExit("Corpus chunk_id is not unique; resolve duplicates before benchmarking.")
print(f"Corpus chunks: {len(corpus):,}; unique questions: {len(questions)}")
print("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

t0 = time.perf_counter()
emb = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
embedding_seconds = time.perf_counter() - t0
emb = np.ascontiguousarray(emb, dtype=np.float32)
q0 = time.perf_counter()
qemb = model.encode(questions["question"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
query_embedding_seconds = time.perf_counter() - q0
qemb = np.ascontiguousarray(qemb, dtype=np.float32)
dim = emb.shape[1]

# Build both indexes with the stated baseline parameters.
t0 = time.perf_counter(); flat = faiss.IndexFlatIP(dim); flat.add(emb); flat_build_s = time.perf_counter()-t0
t0 = time.perf_counter(); hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efConstruction=80; hnsw.hnsw.efSearch=64; hnsw.add(emb); hnsw_build_s = time.perf_counter()-t0
# Serialize indexes to measure their on-disk serialized sizes (not process RSS).
flat_path = RESULTS / "_benchmark_flat.index"; hnsw_path = RESULTS / "_benchmark_hnsw.index"
faiss.write_index(flat, str(flat_path)); faiss.write_index(hnsw, str(hnsw_path))
size_rows = [
 {"method":"Flat","index_build_seconds":flat_build_s,"serialized_index_bytes":flat_path.stat().st_size,"serialized_index_mib":flat_path.stat().st_size/(1024**2),"index_parameters":"IndexFlatIP; normalized embeddings"},
 {"method":"HNSW","index_build_seconds":hnsw_build_s,"serialized_index_bytes":hnsw_path.stat().st_size,"serialized_index_mib":hnsw_path.stat().st_size/(1024**2),"index_parameters":"IndexHNSWFlat; M=32; efConstruction=80; efSearch=64"},
]

# Warm-up each index before timing.
for ix in (flat, hnsw):
    ix.search(qemb[:min(5, len(qemb))], 10)
repetitions = 10
latency_rows=[]; ranking_rows=[]
for rep in range(1, repetitions+1):
    # alternate order to reduce systematic timing bias
    methods = [("Flat",flat),("HNSW",hnsw)] if rep % 2 else [("HNSW",hnsw),("Flat",flat)]
    for method, ix in methods:
        for qi, q in questions.iterrows():
            t0=time.perf_counter(); scores, idxs = ix.search(qemb[qi:qi+1],10); ms=(time.perf_counter()-t0)*1000
            latency_rows.append({"repetition":rep,"question_id":int(q.question_id),"method":method,"search_latency_ms":ms})
            for rank,(score,ixrow) in enumerate(zip(scores[0],idxs[0]), start=1):
                ranking_rows.append({"repetition":rep,"question_id":int(q.question_id),"method":method,"rank":rank,"score":float(score),"chunk_id":ids[int(ixrow)]})

lat=pd.DataFrame(latency_rows); ranks=pd.DataFrame(ranking_rows)
lat.to_csv(RESULTS/"day3_repeated_benchmark_per_query.csv",index=False)
ranks.to_csv(RESULTS/"day3_repeated_benchmark_rankings.csv",index=False)
pd.DataFrame(size_rows).to_csv(RESULTS/"day3_repeated_benchmark_index_sizes.csv",index=False)
summary=[]
for method,g in lat.groupby("method"):
    summary.append({"method":method,"n_timed_searches":len(g),"repetitions":repetitions,"questions":len(questions),"mean_ms":g.search_latency_ms.mean(),"median_ms":g.search_latency_ms.median(),"p95_ms":g.search_latency_ms.quantile(.95),"std_ms":g.search_latency_ms.std(ddof=1),"embedding_corpus_seconds":embedding_seconds,"embedding_queries_seconds":query_embedding_seconds,"index_build_seconds":flat_build_s if method=="Flat" else hnsw_build_s,"note":"Repeated search-only timings after warm-up; embedding and build costs reported separately."})
pd.DataFrame(summary).to_csv(RESULTS/"day3_repeated_benchmark_summary.csv",index=False)

# Compare repeated HNSW rankings with exact Flat rankings per same question and repetition.
flat_r=ranks[ranks.method=="Flat"]; hnsw_r=ranks[ranks.method=="HNSW"]
overlap=[]
for rep in range(1,repetitions+1):
    for qid in questions.question_id:
        a=set(flat_r[(flat_r.repetition==rep)&(flat_r.question_id==qid)].chunk_id)
        b=set(hnsw_r[(hnsw_r.repetition==rep)&(hnsw_r.question_id==qid)].chunk_id)
        overlap.append({"repetition":rep,"question_id":int(qid),"top10_intersection":len(a&b),"top10_jaccard":len(a&b)/len(a|b) if a|b else np.nan,"identical_top10_set":a==b})
ov=pd.DataFrame(overlap)
ov.to_csv(RESULTS/"day3_repeated_benchmark_overlap.csv",index=False)
print("\nDONE")
print(pd.DataFrame(summary).to_string(index=False))
print("Mean per-query top-10 Jaccard:", round(float(ov.top10_jaccard.mean()),4))
print("Questions with identical top-10 sets in every repetition:", int(ov.groupby('question_id').identical_top10_set.all().sum()), "/", len(questions))
print("Output files written under:", RESULTS)
print("Note: these are benchmark outputs, not relevance/ground-truth metrics.")
