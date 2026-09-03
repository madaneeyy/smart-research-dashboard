# Document Retrieval Evaluation — FYP Report

This benchmark is built around the supplied 30-page final-year project report.

It contains 32 questions covering overview, factual, methodology, comparison,
limitations and future-scope queries.

## 1. Prepare the 84 chunks

Export the chunks currently stored for this document into `fyp_chunks.json`:

```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "content": "...",
      "page": 1,
      "document_id": "..."
    }
  ]
}
```

## 2. Run

From the Smart Research Dashboard project root:

```powershell
python evaluate_document_retrieval.py --chunks fyp_chunks.json
```

The script reports:

- Recall@1/3/5/10
- Precision@1/3/5/10
- NDCG@1/3/5/10
- MRR
- query-classification accuracy
- results grouped by query type
- zero-recall questions
- worst questions

## 3. Important

This first version uses **PDF page-level gold evidence**. A result is a hit if
its chunk's `page` is one of the gold pages.

This is a bootstrap benchmark, not the final gold-standard benchmark. Once the
retriever is stable, annotate exact chunk IDs or graded relevance:

- 3 = directly answers/supports
- 2 = strong supporting evidence
- 1 = weak/background relevance
- 0 = irrelevant

Then compare every retriever change using the exact same 32 questions.

Do not tune the retriever until we have a baseline.
