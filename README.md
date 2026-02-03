🧩 Steps Performed by the Code
Step 1: Load the Embedding Model

The pipeline starts by loading a pre-trained Sentence Transformer (all-MiniLM-L6-v2) model.
This model converts text into 384-dimensional semantic embeddings, which capture contextual meaning rather than keywords.

Step 2: Initialize Pinecone Vector Database

A serverless Pinecone index is created (or reused if it already exists) using cosine similarity.
The index is configured to store embeddings efficiently and support low-latency semantic search.

Step 3: Ingest Resume PDFs

All resume PDF files are read from a specified folder.
Text is extracted page-by-page using PyPDF, combined into a single document per resume, and cleaned to remove unnecessary whitespace.

Step 4: Generate Resume Embeddings

Each extracted resume text is passed through the embedding model to generate dense vector representations.
Batch processing is used to improve performance during embedding generation.

Step 5: Store Embeddings in Pinecone

The generated embeddings are upserted into the Pinecone index with unique document IDs.
This makes the resumes searchable using vector similarity.

Step 6: Wait for Indexing Completion

The code continuously checks Pinecone index statistics to ensure all vectors are fully indexed before querying.
This prevents incomplete or inconsistent search results.

Step 7: Execute Semantic Search Queries

Natural-language queries (e.g., required skills or job roles) are embedded using the same model and searched against the Pinecone index to retrieve the top-K most relevant resumes.

Step 8: Benchmark Query Performance

The code measures:

Embedding generation time

Pinecone vector search time

Total end-to-end query latency
