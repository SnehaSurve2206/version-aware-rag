

\# Version-Aware RAG — System Architecture



\## 1. Architecture Overview



The system investigates reliable retrieval from versioned technical documentation using exact and approximate vector search, with version-aware filtering and a proposed selective recovery mechanism.



\## 2. System Architecture Diagram



```mermaid

flowchart TD

&#x20;   A\["Versioned Documentation Corpus<br/>21,052 chunks"] --> B\["Data Preparation<br/>Chunking and Metadata"]

&#x20;   B --> C\["Embedding Model<br/>all-MiniLM-L6-v2"]

&#x20;   C --> D\["Vector Embeddings"]



&#x20;   D --> E\["FAISS Indexes"]

&#x20;   E --> F\["Flat Index<br/>Exact Search"]

&#x20;   E --> G\["HNSW Index<br/>Approximate Search"]



&#x20;   H\["User Query"] --> I\["Query Embedding"]

&#x20;   I --> J{"Target Version Specified?"}



&#x20;   J -->|No| K\["Search Full Corpus"]

&#x20;   J -->|Yes| L\["Version Detection<br/>and Normalization"]

&#x20;   L --> M\["Filter Target-Version Chunks"]



&#x20;   K --> F

&#x20;   K --> G

&#x20;   M --> N\["Filtered Exact Search"]



&#x20;   F --> O\["Retrieved Top-K Evidence"]

&#x20;   G --> O

&#x20;   N --> O



&#x20;   O --> P\["Evaluation"]

&#x20;   P --> Q\["Relevance and Version Correctness"]

&#x20;   P --> R\["Latency and Memory"]

&#x20;   P --> S\["Recall@K and Ranking"]



&#x20;   O -.-> T\["Proposed Selective Recovery"]

&#x20;   T -.-> U\["Re-retrieval / Recovery"]

&#x20;   U -.-> P

```



\## 3. Implemented Components



1\. Documentation corpus preparation.

2\. Sentence Transformer embeddings.

3\. FAISS Flat exact retrieval.

4\. FAISS HNSW approximate retrieval.

5\. Exact target-version filtering.

6\. Retrieval benchmarking and evidence evaluation.



\## 4. Proposed Components



1\. Version-coverage checking.

2\. Evidence-sufficiency checking.

3\. Selective retrieval escalation.

4\. Recovery evaluation and overhead measurement.



These components are research objectives and are not yet fully implemented.



\## 5. Evaluation Outputs



\- Standard Recall@K

\- Correct-version Recall@K

\- Stale-version intrusion rate

\- Retrieval latency

\- Index memory usage

\- Recovery success rate



