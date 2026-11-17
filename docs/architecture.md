VERSION-AWARE RAG — SYSTEM ARCHITECTURE



1\. ARCHITECTURE OVERVIEW



The system investigates reliable retrieval from versioned technical documentation using exact and approximate vector search, version-aware filtering, and a proposed selective retrieval recovery mechanism.



2\. SYSTEM ARCHITECTURE DIAGRAM



The architecture consists of the following workflow:



Versioned Documentation Corpus

→ Data Preparation

→ Embedding Generation

→ FAISS Vector Indexes



User Query

→ Query Embedding

→ Version Detection



If no target version is specified:

→ Full Corpus Retrieval

→ Flat / HNSW Search



If a target version is specified:

→ Version Normalization

→ Target-Version Filtering

→ Filtered Flat Search



Both retrieval paths produce:

→ Retrieved Evidence

→ Evaluation



Evaluation includes:

\- Relevance and Version Correctness

\- Retrieval Latency

\- Index Memory Usage



Proposed Extension:

→ Version-Coverage Checking

→ Evidence-Sufficiency Checking

→ Selective Retrieval Recovery



3\. IMPLEMENTED COMPONENTS



1\. Documentation corpus preparation.

2\. Sentence Transformer embeddings.

3\. FAISS Flat exact retrieval.

4\. FAISS HNSW approximate retrieval.

5\. Initial exact target-version filtering.

6\. Retrieval benchmarking and evidence analysis.



4\. PROPOSED COMPONENTS



1\. Version-coverage checking.

2\. Evidence-sufficiency checking.

3\. Selective retrieval escalation.

4\. Recovery evaluation and overhead measurement.



These components are planned research extensions and are not yet fully implemented.



5\. EVALUATION METRICS



1\. Standard Recall@K.

2\. Correct-Version Recall@K.

3\. Stale-Version Intrusion Rate.

4\. Retrieval Latency.

5\. Index Memory Usage.

6\. Recovery Success Rate.

