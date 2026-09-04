# Smart Research Dashboard

<p align="center">
  <strong>AI-powered research intelligence and Retrieval-Augmented Generation platform</strong>
</p>

<p align="center">
  Discover research • Search technical knowledge • Retrieve evidence • Generate grounded answers
</p>

<p align="center">
  <a href="https://github.com/madaneeyy/smart-research-dashboard">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-FastAPI-009688?logo=fastapi&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/React-TypeScript-3178C6?logo=react&logoColor=white" alt="React TypeScript">
  <img src="https://img.shields.io/badge/GenAI-RAG-7C3AED" alt="RAG">
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?logo=supabase&logoColor=white" alt="Supabase">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Problem](#problem)
- [Goals](#goals)
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [High-Level Architecture](#high-level-architecture)
- [Application Request Flow](#application-request-flow)
- [RAG Architecture](#rag-architecture)
- [Query Classification](#query-classification)
- [Source Acquisition](#source-acquisition)
- [Document Processing](#document-processing)
- [Document Chunking](#document-chunking)
- [Hybrid Retrieval](#hybrid-retrieval)
  - [BM25](#bm25)
  - [TF-IDF](#tf-idf)
  - [Dense Retrieval](#dense-retrieval)
  - [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
- [Reranking](#reranking)
  - [Semantic Relevance](#semantic-relevance)
  - [Metadata Scoring](#metadata-scoring)
  - [Query-Fit Scoring](#query-fit-scoring)
  - [Cross-Encoder Reranking](#cross-encoder-reranking)
- [Evidence Selection](#evidence-selection)
  - [Relevance Filtering](#relevance-filtering)
  - [Duplicate Filtering](#duplicate-filtering)
  - [Maximum Marginal Relevance](#maximum-marginal-relevance)
- [Context Construction](#context-construction)
- [GitHub Repository Retrieval](#github-repository-retrieval)
  - [Repository Representation](#repository-representation)
  - [Repository Tree](#repository-tree)
  - [Query-Aware File Selection](#query-aware-file-selection)
  - [Code Retrieval](#code-retrieval)
- [Research Sources](#research-sources)
- [LLM Integration](#llm-integration)
  - [Groq](#groq)
  - [Ollama](#ollama)
- [Evidence and Source Attribution](#evidence-and-source-attribution)
- [Evaluation](#evaluation)
  - [Evaluation Philosophy](#evaluation-philosophy)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Overall Benchmark](#overall-benchmark)
  - [Performance by Query Type](#performance-by-query-type)
  - [Failure Analysis](#failure-analysis)
  - [Evaluation Workflow](#evaluation-workflow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Local Development](#local-development)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Testing](#testing)
- [Engineering Decisions](#engineering-decisions)
- [Performance and Resource Considerations](#performance-and-resource-considerations)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Development Workflow](#development-workflow)
- [Security](#security)
- [Contributing](#contributing)
- [Project Status](#project-status)
- [License](#license)
- [Author](#author)

---

# Overview

Smart Research Dashboard is a full-stack AI research assistant designed to help users discover, search, analyze, and understand research papers, technical documents, and GitHub repositories.

The application combines traditional information retrieval with semantic retrieval, reranking, evidence selection, and Large Language Models (LLMs).

The central design principle is:

> **Retrieve relevant evidence first, then use an LLM to generate an answer from that context.**

The application is designed to support both research-oriented questions and technical questions that require inspecting source material.

The system currently brings together:

- Research discovery
- Document ingestion
- Document processing
- Document chunking
- Hybrid retrieval
- BM25 retrieval
- TF-IDF retrieval
- Dense semantic retrieval
- Reciprocal Rank Fusion
- Query classification
- Query-aware retrieval
- Metadata-aware ranking
- Query-fit scoring
- Cross-encoder reranking
- Relevance filtering
- Duplicate and near-duplicate filtering
- Maximum Marginal Relevance (MMR)
- Evidence selection
- GitHub repository retrieval
- LLM-powered answer generation
- Evidence and source attribution
- Retrieval evaluation

The overall workflow is:

```text
User Question
      |
      v
Query Analysis
      |
      v
Source Acquisition
      |
      v
Document Processing
      |
      v
Document Chunking
      |
      v
Candidate Retrieval
      |
      v
Candidate Fusion
      |
      v
Reranking
      |
      v
Evidence Selection
      |
      v
Context Construction
      |
      v
LLM Generation
      |
      v
Answer + Supporting Evidence
Problem

Research and technical information is distributed across many different sources.

A typical research workflow can require moving between:

Research papers
Documentation
GitHub repositories
Source code
Technical reports
Uploaded documents
Research databases

Finding the information is often just as difficult as understanding it.

A basic LLM application usually looks like:

Question
   |
   v
LLM
   |
   v
Answer

This does not explicitly solve the retrieval problem.

It does not answer questions such as:

Which source contains the answer?
Which document is relevant?
Which section contains the necessary information?
Which code file contains the implementation?
Which passages actually support the generated response?
How good is the retrieval system independently of the LLM?

Smart Research Dashboard treats retrieval as a first-class part of the application.

Question
   |
   v
Query Understanding
   |
   v
Information Retrieval
   |
   v
Ranking
   |
   v
Reranking
   |
   v
Evidence Selection
   |
   v
LLM
   |
   v
Answer + Evidence
Goals

The project has several engineering goals.

1. Build a practical research assistant

Provide a single environment for discovering, searching, and interacting with scientific and technical information.

2. Go beyond basic vector search

Combine lexical retrieval, semantic retrieval, metadata signals, and reranking instead of depending on one similarity method.

3. Support technical repository research

Allow GitHub repositories to be treated as searchable technical knowledge sources.

4. Preserve evidence

Keep source information attached to retrieved content so that evidence can be inspected after generation.

5. Make retrieval measurable

Use retrieval metrics to evaluate and improve the retrieval pipeline.

6. Keep the architecture modular

Separate:

Source Acquisition
        |
        v
Document Processing
        |
        v
Retrieval
        |
        v
Ranking
        |
        v
Evidence Selection
        |
        v
LLM Generation
        |
        v
Evaluation

so individual components can be modified independently.

Key Features
Research Discovery

The application provides a research-oriented workflow for discovering and analyzing scientific and technical information.

Typical workflow:

Discover Research
        |
        v
Inspect Sources
        |
        v
Search Information
        |
        v
Ask Question
        |
        v
Retrieve Evidence
        |
        v
Generate Answer
Document-Based RAG

Documents can be processed and transformed into retrieval-friendly chunks.

The pipeline retains useful metadata so retrieved information can be associated with its source.

GitHub Repository Research

GitHub repositories can be used as technical knowledge sources.

The system can work with:

Repository metadata
README content
Repository tree
Documentation
Configuration
Source files
Query-focused content

This makes it possible to ask questions such as:

What is this repository about?

Where is this class implemented?

Where is this function defined?

Which files implement this feature?

How does this component work?

How are these components connected?
Hybrid Retrieval

The retrieval system combines:

BM25
TF-IDF
Dense Embeddings
Metadata
Query Fit

before later ranking and evidence-selection stages.

Cross-Encoder Reranking

Initial candidates are reranked using a cross-encoder to provide a more detailed relevance signal.

Query-Aware Retrieval

Queries are classified according to the type of information being requested.

Examples include:

Overview
Facts
Methodology
Results
Comparison
Limitations
Future work
Repository/code questions
Evidence Selection

Candidate retrieval and final evidence selection are separate stages.

The system applies relevance and redundancy filtering and then uses diversity-aware selection to construct the final evidence set.

Evidence Attribution

Retrieved chunks preserve source metadata that can be displayed alongside generated answers.

Screenshots

Application screenshots should be stored in:

screenshots/
├── dashboard.png
├── research-answer.png
├── evidence-panel.png
└── github-retrieval.png
Dashboard

Research Answer

Evidence and Sources

GitHub Repository Retrieval

High-Level Architecture
                         +-------------------------+
                         |    React + TypeScript   |
                         |        Frontend         |
                         +------------+------------+
                                      |
                                  HTTP / SSE
                                      |
                                      v
                         +-------------------------+
                         |        FastAPI          |
                         |         Backend         |
                         +------------+------------+
                                      |
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
      +--------------+        +--------------+        +--------------+
      |   Supabase   |        |   Research   |        |    GitHub    |
      |  PostgreSQL  |        |    Sources   |        |  Retrieval   |
      +--------------+        +------+-------+        +------+-------+
                                     |                       |
                                     +-----------+-----------+
                                                 |
                                                 v
                                  +-------------------------+
                                  | Document Processing     |
                                  |      & Chunking         |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  |    Hybrid Retrieval     |
                                  |                         |
                                  | BM25 + TF-IDF + Dense  |
                                  | Metadata + Query Fit    |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  | Cross-Encoder Reranking |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  | Evidence Selection      |
                                  | Relevance + MMR         |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  |          LLM            |
                                  |      Groq / Ollama      |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  | Answer + Supporting     |
                                  |        Evidence         |
                                  +-------------------------+
Application Request Flow

A typical request moves through the following stages:

1. User submits question
              |
              v
2. FastAPI receives request
              |
              v
3. Request validation
              |
              v
4. Query analysis
              |
              v
5. Query classification
              |
              v
6. Source acquisition
              |
              v
7. Document / repository processing
              |
              v
8. Chunk creation
              |
              v
9. Candidate retrieval
              |
              v
10. Retrieval score fusion
              |
              v
11. Candidate reranking
              |
              v
12. Relevance filtering
              |
              v
13. Duplicate / redundancy filtering
              |
              v
14. Evidence selection
              |
              v
15. Context construction
              |
              v
16. LLM generation
              |
              v
17. Answer + evidence returned
RAG Architecture

The RAG pipeline is intentionally multi-stage.

                         User Question
                              |
                              v
                     Query Classification
                              |
                              v
                      Source Acquisition
                              |
                              v
                    Document Processing
                              |
                              v
                         Chunking
                              |
                              v
                    Candidate Retrieval
                 +------------+------------+
                 |            |            |
                 v            v            v
               BM25         TF-IDF       Dense
                                           |
                                       Embeddings
                 |            |            |
                 +------------+------------+
                              |
                              v
                        Score Fusion
                              |
                              v
                     Candidate Ranking
                              |
                              v
                  Cross-Encoder Reranking
                              |
                              v
                    Relevance Filtering
                              |
                              v
                  Duplicate Filtering
                              |
                              v
                  Evidence / MMR Selection
                              |
                              v
                     Context Construction
                              |
                              v
                             LLM
                              |
                              v
                    Answer + Evidence
Query Classification

The retrieval system distinguishes between different information needs.

For example:

"What is this project about?"

is an overview-oriented question.

"Where is this class implemented?"

is implementation-oriented.

"What were the evaluation results?"

is results-oriented.

The system supports query types including:

Overview
Factual
Methodology
Results
Comparison
Limitation
Future work
Repository/code questions

Query classification provides additional information to downstream retrieval and ranking.

Source Acquisition

Information can enter the system from multiple source types.

                         User Query
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
         Documents          GitHub          Research
                           Repositories       Sources
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Source Collection

The source acquisition layer is kept separate from retrieval so different source types can share the same downstream pipeline.

Document Processing

Documents are processed before retrieval.

Input Document
      |
      v
Text Extraction
      |
      v
Normalization
      |
      v
Content Processing
      |
      v
Chunking
      |
      v
Metadata Preservation
      |
      v
Retrieval

The project includes document-processing support for:

PDF
DOCX
XLSX
PPTX

Libraries used include:

PyMuPDF
pypdf
python-docx
openpyxl
python-pptx
Document Chunking

Large documents are divided into smaller retrieval units.

The purpose of chunking is to:

Improve retrieval precision
Reduce unnecessary context
Preserve localized evidence
Enable ranking at a finer granularity

Chunks can retain metadata such as:

Source
Filename
Section
Page
Chunk Index
Document Type
Repository Path

where applicable.

Hybrid Retrieval

The retrieval layer combines lexical and semantic retrieval methods.

                           Query
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
           BM25            TF-IDF          Dense
         Retrieval        Retrieval      Retrieval
                                             |
                                         Embeddings
             |               |               |
             +---------------+---------------+
                             |
                             v
                        Score Fusion
                             |
                             v
                     Candidate Ranking
BM25

BM25 provides lexical retrieval.

It is useful for queries containing:

Exact terminology
Technical identifiers
File names
Function names
Class names
Domain-specific phrases
TF-IDF

TF-IDF provides an additional lexical retrieval signal.

It complements BM25 by providing another term-based measure of relevance.

Dense Retrieval

Dense retrieval uses sentence-transformer embeddings to identify semantically related content.

Dense retrieval can therefore find content where the exact wording differs from the query.

Reciprocal Rank Fusion

Results from multiple retrieval strategies are combined before later ranking stages.

Conceptually:

BM25 Results
     |
     +------------------+
                        |
TF-IDF Results          |
     |                  |
     +------------------+
                        |
Dense Results           |
     |                  |
     +------------------+
                        |
                        v
                  Rank Fusion
                        |
                        v
                 Candidate Set
Reranking

Initial retrieval is designed to produce a useful candidate set.

The candidates are then evaluated using additional relevance signals.

Candidate Set
     |
     v
Semantic Relevance
     |
     v
Lexical Relevance
     |
     v
Metadata Compatibility
     |
     v
Query-Fit
     |
     v
Cross-Encoder
     |
     v
Final Ranking
Semantic Relevance

Dense similarity provides a semantic relevance signal between the query and candidate chunks.

Metadata Scoring

Metadata can be incorporated when determining how well a candidate matches the query.

Examples include:

Source type
Section
Repository path
Document type
Content category
Query-Fit Scoring

Candidates can be evaluated based on how directly they answer the type of question being asked.

This provides an additional signal beyond raw similarity.

Cross-Encoder Reranking

The cross-encoder receives the query and candidate content together and produces a relevance signal.

Query + Candidate
       |
       v
 Cross-Encoder
       |
       v
 Relevance Score

The cross-encoder is applied after initial retrieval so that detailed scoring is performed over a smaller candidate set.

Evidence Selection

Retrieving candidates is different from selecting final evidence.

The final selection stage can be represented as:

Candidate Retrieval
        |
        v
Candidate Ranking
        |
        v
Relevance Filtering
        |
        v
Duplicate Filtering
        |
        v
Diversity Selection
        |
        v
Final Evidence
Relevance Filtering

Low-value candidates can be removed before final context construction.

This prevents weak evidence from consuming context space.

Duplicate Filtering

Exact or near-duplicate content can occur in retrieval results.

Duplicate filtering helps preserve context capacity for distinct evidence.

Maximum Marginal Relevance

MMR is used to balance:

Relevance
    +
Diversity

The objective is to prefer evidence that is both useful to the query and adds information beyond already-selected chunks.

Context Construction

After final evidence selection, the selected chunks are transformed into LLM context.

Retrieved Evidence
       |
       v
Evidence Selection
       |
       v
Context Formatting
       |
       v
Prompt Construction
       |
       v
LLM

Context can contain:

Source information
File information
Repository paths
Sections
Pages
Retrieved content
GitHub Repository Retrieval

GitHub retrieval is a specialized part of the system for technical repository research.

The goal is to allow users to query repositories without simply treating the entire repository as one large document.

GitHub Repository
       |
       v
Repository Metadata
       |
       v
Repository Tree
       |
       +----------------------+
       |                      |
       v                      v
     README            Documentation
       |                      |
       +----------+-----------+
                  |
                  v
             Configuration
                  |
                  v
              Source Files
                  |
                  v
             Query Analysis
                  |
                  v
        Relevant File Selection
                  |
                  v
        Focused Source Content
                  |
                  v
             RAG Retrieval
                  |
                  v
             LLM Context
Repository Representation

Repository information can be represented using multiple content layers:

Repository metadata
README
Repository tree
Documentation
Configuration
Source files

This provides both high-level and implementation-level information.

Repository Tree

The repository tree provides structural information about available directories and files.

This helps distinguish among:

Source
Documentation
Configuration
Tests
Examples
Other Repository Content
Query-Aware File Selection

The user's question is used to determine which repository content is most relevant.

For example:

"What is this repository about?"

should favor repository-level descriptive information.

While:

"Where is this class implemented?"

should favor implementation-related files.

Code Retrieval

The GitHub workflow is intended to support technical questions involving:

Classes
Functions
Implementations
Related modules
Configuration
Repository structure

Example:

Where is ColumnParallelLinear implemented?

How does this function work?

Which files implement this feature?

How are these modules connected?

The goal is to provide source-level evidence appropriate to the question.

Research Sources

The application is designed to work with multiple research and technical source types.

Current workflows include:

Research documents
arXiv-related research discovery
GitHub repositories
User-provided documents
Technical source material

The source layer is kept separate from retrieval so that additional sources can be integrated without rebuilding the entire system.

LLM Integration

The LLM layer is separated from retrieval.

This allows the application to support different inference approaches.

Current options include:

Groq
Ollama
Groq

The current hosted configuration uses Groq.

LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key
GROQ_MODEL=openai/gpt-oss-120b
Ollama

Local development can use Ollama.

LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b-instruct
LLM Workflow
Question
    |
    v
Retrieved Evidence
    |
    v
Context Construction
    |
    v
Prompt
    |
    v
LLM
    |
    v
Generated Answer
Evidence and Source Attribution

A central objective of the project is to preserve information about the source of retrieved evidence.

Retrieved chunks can contain metadata such as:

Source
Filename
Repository Path
Section
Page
Chunk Index
Document Type
Retrieval Metadata

This allows the application to present both:

Generated Answer

and:

Supporting Evidence

rather than exposing only generated text.

Evaluation

Retrieval quality is evaluated separately from the fluency of generated responses.

The evaluation framework measures whether relevant evidence was retrieved and how well it was ranked.

The current evaluation uses a set of 32 questions over the research report:

A CROSS-DOMAIN STUDY OF ACCURACY, CALIBRATION,
AND ROBUSTNESS IN CNN, TRANSFORMER, AND
SEQUENTIAL VISION MODELS

The evaluation contains multiple query categories, including:

Overview
Factual
Methodology
Comparison
Limitation
Future-oriented questions
Evaluation Philosophy

A RAG system should not be evaluated only by asking:

"Does the generated answer sound good?"

The retrieval system should also be evaluated directly.

The evaluation process therefore asks:

Did we retrieve the relevant evidence?

How highly was it ranked?

How much relevant evidence was retrieved?

Was the query classified correctly?

Which questions remain difficult?

The development loop is:

Identify Failure
      |
      v
Inspect Retrieved Evidence
      |
      v
Determine Retrieval Problem
      |
      v
Modify Retrieval / Ranking
      |
      v
Run Benchmark
      |
      v
Compare Metrics
      |
      v
Inspect Failures Again
Evaluation Metrics
Metric	Description
Recall@K	Measures whether relevant evidence appears within the top K results
Precision@K	Measures how much of the retrieved set is relevant
MRR	Measures how early the first relevant result appears
nDCG@K	Measures ranking quality while accounting for position
Classification Accuracy	Measures query classification correctness
Overall Benchmark

The current evaluation contains:

32 Questions

Overall results:

Metric	Score
MRR	0.5133
Classification Accuracy	1.0000
Recall@1	0.2057
Precision@1	0.3438
nDCG@1	0.3438
Recall@3	0.3719
Precision@3	0.2813
nDCG@3	0.3786
Recall@5	0.5073
Precision@5	0.2396
nDCG@5	0.4470
Recall@10	0.7078
Precision@10	0.2396
nDCG@10	0.5321

These figures are from the current internal development benchmark.

The benchmark is used to compare retrieval configurations and identify weaknesses in retrieval and ranking.

Performance by Query Type

The benchmark also reports retrieval performance by query category.

Query Type	Questions	Recall@5	Recall@10	Precision@5	MRR
Comparison	3	0.4444	0.8889	0.1333	0.2500
Factual	13	0.6859	0.6859	0.2821	0.7692
Limitation	1	0.3333	1.0000	0.2000	0.5000
Methodology	6	0.5000	0.6667	0.2333	0.3250
Overview	9	0.2944	0.6741	0.2222	0.3585

The classification accuracy is 100% across the benchmark, while retrieval quality varies substantially by query type.

The factual questions currently have the strongest MRR and Recall@5 performance, while overview and methodology questions remain more challenging.

Failure Analysis

The evaluation is also used to identify individual retrieval failures rather than only reporting aggregate metrics.

The current benchmark identifies the following questions with zero Recall@10:

FACT04
FACT05
METH06

These correspond to:

FACT04
What evaluation metrics were used?

FACT05
What corruptions were used for robustness testing?

METH06
How was robustness evaluated?

The benchmark also identifies several difficult cases at Recall@5, including:

FACT04
FACT05
METH06
OV04
OV05
METH02
RES03
RES04
FUT01
FACT10

These failures are useful during development because they reveal specific retrieval weaknesses rather than simply indicating that the overall score changed.

Example Evaluation Cases
Overview Query
What is this report about?

The benchmark reports:

First relevant rank: 2
MRR: 0.50
Recall@5: 0.6667
Recall@10: 0.6667

Factual Query
What datasets were used in the project?

The benchmark reports:

First relevant rank: 1
MRR: 1.00
Recall@5: 0.3333
Recall@10: 0.3333

Methodology Query
How was the experimental pipeline standardized across datasets and models?

The benchmark reports:

First relevant rank: 1
MRR: 1.00
Recall@5: 1.00
Recall@10: 1.00

Difficult Implementation-Oriented Methodology Query
How was ResNet-18 implemented?

The benchmark reports:

First relevant rank: 6
MRR: 0.1667
Recall@5: 0.0
Recall@10: 1.0

This shows an important distinction between:

Finding relevant evidence somewhere in the top 10

and:

Ranking the relevant evidence near the top

Evaluation Workflow

The evaluation process is designed to support iterative RAG development.

                 Benchmark Dataset
                        |
                        v
                Run Retriever
                        |
                        v
                Measure Metrics
                        |
             +----------+----------+
             |                     |
             v                     v
        Aggregate              Per-query
         Metrics               Analysis
             |                     |
             +----------+----------+
                        |
                        v
                 Failure Analysis
                        |
                        v
                Retrieval Changes
                        |
                        v
                  Re-run Test

The benchmark can therefore be used to compare different retrieval implementations and ranking strategies.

Technology Stack
Backend
Python
FastAPI
Pydantic
REST APIs
Server-Sent Events (SSE)
Generative AI
Groq
Ollama
Large Language Models
Prompt Engineering
Retrieval-Augmented Generation
Retrieval
BM25
TF-IDF
Sentence Transformers
Dense Embeddings
Reciprocal Rank Fusion
Cross-Encoder Reranking
Metadata-Aware Ranking
Query-Fit Scoring
Relevance Filtering
Duplicate Filtering
Maximum Marginal Relevance
Document Processing
PyMuPDF
pypdf
python-docx
openpyxl
python-pptx
Database
Supabase
PostgreSQL
Frontend
React
TypeScript
Vite
Development and Evaluation
Git
GitHub
Automated Tests
Smoke Tests
Retrieval Evaluation
Project Structure
smart-research-dashboard/
│
├── app/
│
├── backend/
│   ├── db/
│   ├── routes/
│   ├── services/
│   └── main.py
│
├── src/
│   ├── clients/
│   ├── collectors/
│   ├── models/
│   └── services/
│       ├── document_rag/
│       ├── github/
│       ├── rag/
│       └── ...
│
├── evaluation/
│
├── frontend/
│
├── tests/
│
├── screenshots/
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── github_retrieval_eval.py
├── pwc_smoke_test.py
├── requirements-dev.txt
├── requirements.txt
└── smoke_test.py
Local Development
Prerequisites

Install:

Python 3.10+
Node.js
npm
Git

Optional:

Ollama for local LLM inference
Installation
Clone the Repository
git clone https://github.com/madaneeyy/smart-research-dashboard.git
cd smart-research-dashboard
Create a Python Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
Install Backend Dependencies
pip install -r requirements.txt

For development dependencies:

pip install -r requirements-dev.txt
Environment Variables

Create a local .env file using .env.example as a reference.

Example:

# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key
GROQ_MODEL=openai/gpt-oss-120b

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

For local Ollama:

LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b-instruct

Never commit real credentials.

Running the Backend

From the repository root:

uvicorn backend.main:app --reload
Running the Frontend

Open another terminal:

cd frontend
npm install
npm run dev
Testing

Application and retrieval validation utilities are included in the repository.

Application Smoke Test
python smoke_test.py
Additional Smoke Test
python pwc_smoke_test.py
GitHub Retrieval Evaluation
python github_retrieval_eval.py

Additional evaluation utilities are available under:

evaluation/
Engineering Decisions
Retrieval Before Generation

The application separates retrieval and generation.

Retrieve
   |
   v
Select Evidence
   |
   v
Generate

This makes retrieved context an explicit component of the application.

Hybrid Retrieval

Lexical and semantic retrieval are combined because they provide different retrieval signals.

Lexical retrieval is useful for exact terminology and identifiers.

Dense retrieval is useful for semantic similarity.

Multi-Stage Ranking

Candidate retrieval is separated from evidence selection.

Candidate Retrieval
        |
        v
Candidate Ranking
        |
        v
Evidence Selection

This allows the retrieval system to explore a broader candidate set before constructing the final context.

Query-Aware Retrieval

Different questions require different evidence.

Overview
   !=
Facts
   !=
Methodology
   !=
Results
   !=
Implementation

Query classification provides an additional signal for retrieval.

Evidence Selection

The system does not assume that the first retrieved result is always sufficient.

The final context can consider:

Relevance
Query fit
Metadata
Redundancy
Diversity
Complementarity
Evaluation-Driven Development

Retrieval changes are measured using benchmark metrics.

This makes it possible to determine whether a retrieval change actually improves the system.

Performance and Resource Considerations

Dense embedding and reranking models can use substantially more memory than the base API application.

The project therefore includes resource-aware optimizations.

These include:

Lazy loading of heavy components where possible
Single-worker operation for constrained environments
Shared embedding model usage where possible
Candidate-size limits
Controlled context sizes
Releasing heavy retrieval models where appropriate

These optimizations are particularly relevant when the backend is deployed to resource-constrained environments.

Current Limitations

The project is actively being developed.

Retrieval Quality

Some questions still produce suboptimal evidence sets.

The evaluation shows that performance varies by question type.

For example:

Factual queries currently have strong performance.
Overview queries are more difficult.
Some methodology queries retrieve relevant evidence only at lower ranks.
Some robustness-related questions remain difficult.
GitHub Code Retrieval

Implementation-level repository queries require precise retrieval of:

Classes
Functions
Definitions
Related modules
Supporting files

Improving source-level retrieval remains an active development area.

Evidence Grounding

Retrieving relevant evidence does not automatically guarantee that every generated claim is explicitly supported by that evidence.

Improving:

Retrieved Evidence
        |
        v
Generated Claims

remains an important area of development.

Resource Usage

Embedding and reranking models increase memory usage compared with the basic API layer.

Memory-aware loading and candidate controls are therefore important for deployment.

Deployment

Deployment performance and reliability remain active engineering areas, particularly in constrained compute and memory environments.

Roadmap
Retrieval
Improve code-aware GitHub retrieval
Improve repository overview retrieval
Improve query-specific evidence selection
Improve ranking signals
Improve evidence diversity
Reduce irrelevant retrievals
Evaluation
Expand the benchmark dataset
Add more retrieval failure categories
Add evidence-grounding evaluation
Add more ranking experiments
Compare retrieval configurations systematically
Backend
Improve observability
Improve performance
Continue reducing memory usage
Expand automated testing
Improve deployment workflows
Research Sources
Expand supported research sources
Improve source normalization
Improve metadata extraction
Improve research discovery
Developer Experience
Improve documentation
Improve reproducibility
Add CI/CD workflows
Improve development tooling
Development Workflow

The project follows an iterative development process.

Implement
    |
    v
Test
    |
    v
Benchmark
    |
    v
Inspect Failures
    |
    v
Improve
    |
    v
Benchmark Again

For RAG changes, both aggregate metrics and individual failure cases are considered.

This is important because a higher-level metric alone may not explain why individual questions succeed or fail.

Security

Never commit:

API keys
Supabase credentials
Authentication tokens
Private credentials
.env files containing secrets
Sensitive documents

Use environment variables for secrets.

Recommended repository configuration:

.env.example
.gitignore

The .env.example file should contain placeholders only.

Contributing

The project is primarily maintained as a personal engineering and research project.

For significant changes:

Create Branch
      |
      v
Implement Change
      |
      v
Run Tests
      |
      v
Run Relevant Evaluation
      |
      v
Review Change
      |
      v
Commit

When changing retrieval behavior, benchmark the change before and after the modification whenever practical.

Project Status

Active Development

The project currently includes:

FastAPI backend
React + TypeScript frontend
Supabase integration
Groq LLM integration
Ollama local inference
Document processing
Document chunking
BM25 retrieval
TF-IDF retrieval
Dense semantic retrieval
Hybrid retrieval
Reciprocal Rank Fusion
Cross-encoder reranking
Query classification
Metadata-aware ranking
Query-fit scoring
Relevance filtering
Duplicate filtering
MMR-based evidence selection
GitHub repository retrieval
Research discovery
Evidence/source attribution
Retrieval evaluation
Application smoke tests

Current development focuses on:

Improving retrieval quality
Improving GitHub code retrieval
Improving repository overview retrieval
Improving evidence grounding
Improving ranking quality
Optimizing memory usage
Reducing retrieval latency
Expanding evaluation coverage
Improving deployment reliability
License

This project is licensed under the MIT License.

See the LICENSE file for details.

Author

Madan Pandey

B.Tech Computer Science & Engineering

GitHub:

https://github.com/madaneeyy

Project:

https://github.com/madaneeyy/smart-research-dashboard

Acknowledgements

This project builds on open-source tools and libraries from the Python, FastAPI, React, Hugging Face, Supabase, Groq, Ollama, and broader open-source communities.
