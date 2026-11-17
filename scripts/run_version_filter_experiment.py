
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(".")
DATA = ROOT / "data" / "chunks" / "chunks_export.csv"
BASELINE = ROOT / "results" / "day3_baseline_top10.csv"
OUT = ROOT / "results" / "day4_version_filter_comparison.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10

# Q16's target is provisional pending a cross-version link audit.
CASES = {
    16: {
        "target_version": "120rc0",
        "evidence_id": "d39296bb-8259-571d-b205-b336824f4582",
    },
    19: {
        "target_version": "120rc7",
        "evidence_id": "15f940a0-0250-5720-b1a7-3b471e5455f0",
    },
    42: {
        "target_version": "120rc7",
        "evidence_id": "43d2df97-d049-5eac-a29f-d8faeb2caf1f",
    },
}

df = pd.read_csv(DATA).fillna("")
baseline = pd.read_csv(BASELINE)

df["version_normalized"] = (
    df["version_raw"].astype(str).str.strip().str.lower()
)
df["text"] = df["text"].astype(str)

# Read one canonical question per ID from the existing baseline.
questions = (
    baseline[["question_id", "question"]]
    .drop_duplicates("question_id")
)
questions["question_id"] = questions["question_id"].astype(int)
question_map = questions.set_index("question_id")["question"].to_dict()

model = SentenceTransformer(MODEL_NAME)

# Encode the corpus once, using the same normalized embeddings for both methods.
start = time.perf_counter()
corpus_embeddings = model.encode(
    df["text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype("float32")
corpus_encode_seconds = time.perf_counter() - start

# Full-corpus exact cosine-similarity index.
full_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
full_index.add(corpus_embeddings)

rows = []

for qid, case in CASES.items():
    if qid not in question_map:
        raise ValueError(f"Question Q{qid} missing from baseline CSV")

    question = question_map[qid]
    target = case["target_version"]
    evidence_id = case["evidence_id"]

    qvec = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Method 1: search the complete corpus.
    start = time.perf_counter()
    scores, indices = full_index.search(qvec, TOP_K)
    full_latency_ms = (time.perf_counter() - start) * 1000

    full_results = []
    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx < 0:
            continue
        full_results.append((rank, float(score), int(idx)))

    # Method 2: filter by exact target version, then run exact search.
    subset_positions = np.flatnonzero(
        df["version_normalized"].to_numpy() == target
    )
    if len(subset_positions) == 0:
        raise ValueError(f"No corpus rows for target version {target}")

    subset_embeddings = corpus_embeddings[subset_positions]
    subset_index = faiss.IndexFlatIP(subset_embeddings.shape[1])
    subset_index.add(subset_embeddings)

    start = time.perf_counter()
    f_scores, f_indices = subset_index.search(
        qvec, min(TOP_K, len(subset_positions))
    )
    filtered_latency_ms = (time.perf_counter() - start) * 1000

    filtered_results = []
    for rank, (score, local_idx) in enumerate(
        zip(f_scores[0], f_indices[0]), start=1
    ):
        if local_idx < 0:
            continue
        global_idx = int(subset_positions[int(local_idx)])
        filtered_results.append((rank, float(score), global_idx))

    for method, results, latency in [
        ("Flat_unfiltered", full_results, full_latency_ms),
        ("Flat_version_filtered", filtered_results, filtered_latency_ms),
    ]:
        evidence_rank = next(
            (
                rank
                for rank, _, idx in results
                if str(df.iloc[idx]["chunk_id"]) == evidence_id
            ),
            None,
        )

        for rank, score, idx in results:
            row = df.iloc[idx]
            rows.append({
                "question_id": qid,
                "question": question,
                "target_version": target,
                "target_is_provisional": qid == 16,
                "method": method,
                "rank": rank,
                "score": score,
                "chunk_id": row["chunk_id"],
                "version_raw": row["version_raw"],
                "document": row["document"],
                "designated_evidence_rank": evidence_rank,
                "designated_evidence_in_top10": evidence_rank is not None,
                "query_latency_ms": latency,
                "corpus_embedding_seconds": corpus_encode_seconds,
            })

    print(
        f"Q{qid} | target={target} | "
        f"full evidence rank={next((r for r, _, i in full_results if str(df.iloc[i]['chunk_id']) == evidence_id), None)} | "
        f"filtered evidence rank={next((r for r, _, i in filtered_results if str(df.iloc[i]['chunk_id']) == evidence_id), None)}"
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)

print(f"\nCorpus embedding time: {corpus_encode_seconds:.2f}s")
print(f"Saved detailed results to: {OUT}")
print("Note: timings are single-run measurements, not a statistical benchmark.")
