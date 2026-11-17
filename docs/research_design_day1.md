Day 1 — Research Design



WORKING TITLE



Reliable Version-Aware Retrieval in RAG Systems: Evaluating Approximate Vector Search and Selective Retrieval Recovery



RESEARCH QUESTION



How do approximate and quantized vector search methods affect correct-version retrieval when the requested version is ambiguous, and can selective recovery improve reliability while controlling latency and memory?



HYPOTHESES



H1: Some approximate or compressed indexes may miss correct-version evidence.



H2: Version-coverage checks may detect failures missed by generic relevance checks.



H3: Selective recovery may improve reliability without excessive overhead.



These are hypotheses to test, not established findings.



QUERY CATEGORIES



1\. Named-version questions.

2\. Unspecified-version questions.

3\. Cross-version questions.



BASELINES



1\. Exact Flat search.

2\. Mixed-version HNSW search.

3\. Per-version partitioned search.

4\. Generic evidence-sufficiency checking, if feasible.

5\. Proposed version-coverage and recovery method.

6\. Quantized search configurations, where supported.



EVALUATION METRICS



1\. Correct-version Recall@k.

2\. Stale-version intrusion rate.

3\. Standard Recall@k.

4\. Answer accuracy, if generation is evaluated.

5\. Retrieval latency.

6\. Index memory usage.

7\. Recovery success rate.



EXPERIMENTAL CONTROLS



1\. Keep the embedding model, chunks, distance metric and queries consistent.

2\. Record software versions and index parameters.

3\. Separate development, validation and held-out test data.

4\. Tune thresholds without using test results.

5\. Verify evidence and version labels.

6\. Treat legitimate historical evidence as correct for historical questions.

7\. Report negative and null results honestly.



HARDWARE AND ENVIRONMENT



\- Windows laptop.

\- Python 3.12 virtual environment.

\- Approximately 16 GB RAM.

\- Intel Iris Xe integrated graphics.

\- FAISS, Sentence Transformers, pandas and matplotlib installed.



DAY 2 TASKS



1\. Verify dataset availability and licensing.

2\. Select suitable versioned documents.

3\. Design controlled version edits.

4\. Verify supported FAISS index configurations.

5\. Create and validate benchmark questions.





PROGRESS UPDATE — NOVEMBER 2026



COMPLETED



1\. Prepared a corpus containing 21,052 documentation chunks.

2\. Created a 50-question evaluation set.

3\. Implemented exact Flat and HNSW retrieval baselines.

4\. Conducted repeated benchmarking with 10 repetitions.

5\. Analyzed retrieval overlap and identified five affected questions.

6\. Audited evidence for selected version-specific questions.

7\. Implemented an initial exact-version-filtered retrieval experiment.



PRELIMINARY FINDINGS



\- Flat mean search latency: 6.64 ms.

\- HNSW mean search latency: 0.92 ms.

\- HNSW showed lower search latency in the current benchmark.

\- Exact-version filtering retrieved designated evidence at ranks 2, 1 and 1 for Q16, Q19 and Q42 respectively.

\- These findings are preliminary and require broader evaluation.



IN PROGRESS



1\. Validate version-specific ground-truth labels.

2\. Evaluate version-filtered retrieval across the full benchmark.

3\. Implement selective version-aware retrieval and recovery.

4\. Compare retrieval quality, latency and memory overhead.

