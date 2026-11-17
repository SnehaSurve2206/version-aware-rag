EXPERIMENTAL RESULTS — VERSION-AWARE RAG



1\. EXPERIMENTAL OVERVIEW



The experiments investigate the effect of exact and approximate vector search on retrieval performance across versioned technical documentation.



The initial evaluation uses:



\- 21,052 documentation chunks.

\- 50 evaluation questions.

\- all-MiniLM-L6-v2 embedding model.

\- FAISS Flat and HNSW indexes.



2\. BASELINE RETRIEVAL BENCHMARK



Two retrieval methods were evaluated:



1\. Flat Search — Exact nearest-neighbor search.

2\. HNSW Search — Approximate nearest-neighbor search.



EXPERIMENTAL CONFIGURATION



\- Embedding model: all-MiniLM-L6-v2

\- Similarity metric: Inner Product with normalized embeddings

\- HNSW M: 32

\- HNSW efConstruction: 80

\- HNSW efSearch: 64

\- Evaluation questions: 50

\- Repetitions: 10

\- Timed searches per method: 500



PERFORMANCE RESULTS



Metric: Mean latency

Flat: 6.64 ms

HNSW: 0.92 ms



Metric: Median latency

Flat: 5.17 ms

HNSW: 0.62 ms



Metric: P95 latency

Flat: 16.31 ms

HNSW: 2.02 ms



Metric: Standard deviation

Flat: 4.10 ms

HNSW: 1.46 ms



Metric: Index build time

Flat: 0.10 seconds

HNSW: 5.40 seconds



Metric: Serialized index size

Flat: 30.84 MiB

HNSW: 36.30 MiB



OBSERVATIONS



\- HNSW achieved approximately 7.23 times lower mean search latency.

\- HNSW also showed lower P95 search latency.

\- HNSW required approximately 5.30 seconds more index-building time.

\- HNSW's serialized index was approximately 17.7% larger than Flat.



These measurements represent one benchmark run containing 10 repeated measurements per method, not 10 independent experiments.



3\. RETRIEVAL OVERLAP ANALYSIS



Flat and HNSW top-10 retrieval results were compared across 50 questions and 10 repetitions.



RESULTS



\- Total comparisons: 500

\- Identical top-10 result sets: 450

\- Non-identical top-10 result sets: 50

\- Mean Jaccard similarity: 0.9606



Five questions showed differing top-10 result sets across all 10 repetitions:



\- Q16

\- Q19

\- Q26

\- Q30

\- Q42



OBSERVATION



Although Flat and HNSW produced highly overlapping retrieval results overall, selected questions demonstrated differences in retrieved evidence.



High retrieval overlap does not establish retrieval correctness.



4\. INITIAL VERSION-FILTERED RETRIEVAL EXPERIMENT



An initial experiment compared full-corpus Flat retrieval against exact target-version-filtered Flat retrieval.



Three selected questions were evaluated.



QUESTION Q16



Target Version: 120rc0

Full Flat Evidence Rank: Not in top 10

Filtered Flat Evidence Rank: 2



QUESTION Q19



Target Version: 120rc7

Full Flat Evidence Rank: Not in top 10

Filtered Flat Evidence Rank: 1



QUESTION Q42



Target Version: 120rc7

Full Flat Evidence Rank: Not in top 10

Filtered Flat Evidence Rank: 1



OBSERVATIONS



\- Designated evidence chunks were retrieved in all three filtered searches.

\- None of the designated evidence chunks appeared in the corresponding unfiltered top-10 results.

\- Exact-version filtering improved the rank of the designated evidence in these selected cases.



Q16's target version is provisional. The first version containing the relevant documentation change requires further verification.



LIMITATIONS



\- Only three questions were evaluated.

\- Search timings were measured in a single run.

\- Full benchmark evaluation has not yet been completed.

\- Retrieval relevance and version correctness require broader validation.



5\. CURRENT FINDINGS



The initial experiments indicate:



1\. HNSW can substantially reduce search latency

