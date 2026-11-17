VERSION-AWARE RETRIEVAL-AUGMENTED GENERATION (RAG)



Reliable Version-Aware Retrieval in RAG Systems: Evaluating Approximate Vector Search and Selective Retrieval Recovery



OVERVIEW



Traditional Retrieval-Augmented Generation (RAG) systems retrieve documents based primarily on semantic similarity. However, when documentation contains multiple software versions, semantically similar results may belong to an incorrect version.



This project investigates how version-aware retrieval can improve the reliability of retrieving version-specific information.



The research evaluates exact and approximate vector search methods and investigates selective retrieval recovery mechanisms.



RESEARCH OBJECTIVES



1\. Evaluate the effect of approximate vector search on version-correct retrieval.

2\. Investigate version-specific evidence retrieval.

3\. Compare exact Flat search and approximate HNSW search.

4\. Develop version-aware filtering and selective retrieval recovery.

5\. Measure retrieval quality, latency and index memory usage.



DATASET



Total documentation chunks: 21,052

Evaluation questions: 50

Domain: Versioned technical documentation

Embedding model: all-MiniLM-L6-v2



EXPERIMENTAL METHODS



1\. Flat Search: Exact vector similarity search.

2\. HNSW Search: Approximate nearest-neighbor retrieval.

3\. Version-Filtered Retrieval: Restricts retrieval to the requested software version.

4\. Version-Aware Escalation: Planned mechanism for detecting and recovering from retrieval failures.



PRELIMINARY RESULTS



Flat vs HNSW Benchmark



Metric                  Flat             HNSW

Mean search latency     6.64 ms          0.92 ms

P95 latency             16.31 ms         2.02 ms

Index size              30.84 MiB        36.30 MiB



HNSW demonstrated lower search latency in the current benchmark, with a larger serialized index.



INITIAL VERSION-FILTERING EXPERIMENT



Question       Target Version       Filtered Evidence Rank

Q16            120rc0               2

Q19            120rc7               1

Q42            120rc7               1



The designated evidence chunks were retrieved in all three filtered searches. These are preliminary results from a small experiment, not a final evaluation.



EVALUATION METRICS



1\. Correct-version Recall@k

2\. Standard Recall@k

3\. Stale-version intrusion rate

4\. Retrieval latency

5\. Index memory usage

6\. Recovery success rate



TECHNOLOGY STACK



Python 3.12

FAISS

Sentence Transformers

NumPy

Pandas

Matplotlib



PROJECT STRUCTURE



version-aware-rag/



&#x20;   data/

&#x20;       Local datasets (not tracked)



&#x20;   docs/

&#x20;       Research documentation



&#x20;   results/

&#x20;       Experimental results



&#x20;   scripts/

&#x20;       run\_repeated\_benchmark.py

&#x20;       run\_version\_filter\_experiment.py



&#x20;   src/

&#x20;       day3\_baseline.py



&#x20;   .gitignore

&#x20;   README.md



CURRENT RESEARCH STATUS



Dataset preparation: Completed

Baseline retrieval: Completed

Repeated Flat and HNSW benchmarking: Completed

Initial version-filtering experiment: Completed

Full version-aware evaluation: In progress

Selective retrieval escalation: Planned

Final comparative evaluation: Pending



LIMITATIONS



1\. Initial version-filtering results cover only three selected questions.

2\. Q16's target version is provisional and requires further verification.

3\. Evaluation labels require additional validation.

4\. Full-scale retrieval quality and latency comparisons are pending.



FUTURE WORK



1\. Complete version-specific ground-truth validation.

2\. Evaluate version-filtered retrieval across the benchmark.

3\. Implement selective retrieval escalation.

4\. Compare all retrieval approaches.

5\. Conduct ablation studies.

6\. Prepare the final research paper.



AUTHOR



Sneha Gajanan Surve



B.Tech — Artificial Intelligence and Machine Learning



GitHub: https://github.com/SnehaSurve2206

