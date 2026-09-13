# Bujha AI — Smart Research Dashboard

> An AI-powered research workspace for discovering, organizing, retrieving, and querying technical knowledge across research papers, GitHub repositories, ML models, and uploaded documents.

**Live Demo:** https://bujha.vercel.app/  
**GitHub:** https://github.com/madaneeyy/smart-research-dashboard

---

## Overview

Bujha AI is a full-stack AI research assistant designed to help users find, organize, and query technical information from multiple research sources in a single workspace.

Rather than functioning as a generic chatbot, the system combines **research discovery, document processing, repository retrieval, hybrid information retrieval, evidence ranking, and grounded LLM generation**.

The application supports research workflows across:

- Research papers
- GitHub repositories and source code
- Machine learning models
- Uploaded documents
- Persistent research workspaces
- Research-focused conversations

The system is built around a retrieval pipeline that combines **semantic retrieval and lexical BM25 search**, followed by rank fusion, relevance filtering, reranking, and evidence selection before the final response is generated.

---

## Key Features

### Multi-Source Research Discovery

Search and discover technical resources from:

- **arXiv** — research papers and metadata
- **GitHub** — repositories and repository content
- **PapersWithCode** — papers, methods, datasets, and paper-code relationships
- **Hugging Face** — models and ML resources

Provider-specific results are normalized into a common research representation so heterogeneous sources can be handled consistently.

---

### Workspace-Based Research

Research is organized into persistent workspaces.

A workspace can contain:

- Sources
- Documents
- Papers
- GitHub repositories
- Models
- Chats
- Recent activity

This allows related research to remain organized rather than being spread across isolated searches and conversations.

---

### AI-Powered Research Chat

Users can ask research questions against selected sources and documents.

The chat system supports:

- Persistent conversations
- Source attachments
- Document attachments
- Streamed responses
- Retrieval metadata
- Evidence display
- Grounded responses

The intended flow is:

```text
User Question
     ↓
Query Classification / Routing
     ↓
Retrieve Candidate Evidence
     ↓
Rank and Filter
     ↓
Select Supporting Evidence
     ↓
Build Grounded Context
     ↓
LLM Generation
     ↓
Response + Evidence
```

---

### Document Processing

Bujha AI supports ingestion of:

- PDF
- DOCX
- XLSX
- PPTX

Uploaded content is processed into retrieval-friendly chunks while preserving useful structural information such as:

- Headings
- Section hierarchy
- Parent sections
- Section paths
- Page information
- Chunk indices
- Chunk type
- File paths
- Programming language where applicable

---

### GitHub Repository Retrieval

GitHub repositories can be incorporated into the research workflow.

Instead of sending entire repositories directly to the language model, repository information and source files are acquired, filtered, chunked, and ranked before relevant content is selected.

This is especially useful for questions such as:

- What does this repository do?
- How is a particular feature implemented?
- Where is a specific class or function defined?
- How does one implementation compare with another?
- What technical approach does the repository use?

The repository workflow also uses caching and fallback acquisition paths to reduce unnecessary repeated retrieval.

---

# Architecture

```text
                         ┌─────────────────────────────┐
                         │      React + TypeScript      │
                         │          Frontend            │
                         └──────────────┬──────────────┘
                                        │
                               HTTP / JSON / Streaming
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │        FastAPI Backend       │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
      ┌──────────────┐         ┌────────────────┐        ┌────────────────┐
      │ Workspace    │         │ Research       │        │ Chat / Activity│
      │ APIs         │         │ APIs           │        │ APIs           │
      └──────────────┘         └────────────────┘        └────────────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │       Research / RAG        │
                         │           Layer             │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
      ┌──────────────┐         ┌────────────────┐        ┌────────────────┐
      │ Semantic     │         │ BM25 Lexical   │        │ Query Routing  │
      │ Retrieval    │         │ Retrieval      │        │ & Relevance    │
      └──────┬───────┘         └───────┬────────┘        └────────────────┘
             │                         │
             └──────────────┬──────────┘
                            ▼
                    ┌──────────────────┐
                    │ Reciprocal Rank  │
                    │ Fusion (RRF)     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Relevance        │
                    │ Filtering        │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ CrossEncoder     │
                    │ Reranking        │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ MMR Evidence     │
                    │ Selection        │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ LLM Generation   │
                    └────────┬─────────┘
                             ▼
                    Grounded Response
```

---

# Retrieval Pipeline

One of the main engineering components of Bujha AI is its hybrid retrieval pipeline.

The system does not rely on a single retrieval strategy.

Instead, it combines multiple retrieval stages to improve technical search quality.

---

## 1. Query Classification and Routing

Queries are classified into research-oriented categories including:

- Factual
- Methodology
- Overview
- Comparison
- Limitation
- Cross-source / relationship queries

The classification and routing layer determines how the query should be handled by downstream research and retrieval components.

---

## 2. Semantic Retrieval

Dense semantic retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Semantic retrieval is useful when the relevant evidence uses different wording from the original query but expresses a similar concept.

The embedding layer supports:

- Local Sentence Transformer inference
- Remote Hugging Face inference

This allows embedding generation to be separated from the application infrastructure.

---

## 3. BM25 Retrieval

BM25 provides lexical retrieval for exact terms and technical identifiers.

This is particularly useful for:

- Model names
- Function names
- Class names
- File paths
- API names
- Repository-specific terminology
- Technical keywords

The searchable representation can also incorporate structural metadata such as:

```text
path
category
section
parent_section
section_path
chunk_type
language
symbol
```

---

## 4. Reciprocal Rank Fusion

Semantic and lexical result rankings are combined using **Reciprocal Rank Fusion (RRF)**.

```text
Semantic Search ───────┐
                       ├──> RRF ───> Unified Candidate Ranking
BM25 Search ───────────┘
```

This allows the system to benefit from both semantic similarity and exact lexical matching.

---

## 5. Relevance Filtering

Candidate results are filtered before final evidence selection.

The system applies relevance scoring and removes exact duplicate results so that low-value or repeated evidence is less likely to reach the generation stage.

---

## 6. CrossEncoder Reranking

For the document RAG path, candidate results can be reranked using a CrossEncoder.

This provides a second-stage query-aware ranking step after the initial retrieval process.

---

## 7. MMR-Based Evidence Selection

Maximum Marginal Relevance is used to reduce redundant evidence.

The selection process prioritizes:

```text
Relevance
    ↓
Exact technical matching
    ↓
Complementary supporting evidence
    ↓
Diversity
```

Highly relevant primary evidence is protected so that diversity does not unnecessarily replace the strongest evidence.

---

# Structure-Aware Chunking

Bujha AI uses structure-aware document and code chunking rather than relying only on naive fixed-size text splitting.

The chunking system preserves meaningful structure including:

- Headings
- Heading hierarchy
- Paragraphs
- Lists
- Fenced code blocks
- Section paths
- Parent sections
- File paths
- Language
- Chunk type
- Chunk indices

Oversized content can be split further while attempting to preserve meaningful sentence boundaries.

This structural information becomes useful during lexical search, relevance scoring, and evidence selection.

---

# Research Discovery

Bujha AI integrates multiple external research ecosystems.

| Source | Role |
|---|---|
| arXiv | Research paper discovery and metadata |
| GitHub | Repository discovery and source-code retrieval |
| PapersWithCode | Papers, methods, datasets, and paper-code relationships |
| Hugging Face | Model and ML resource discovery |

Results from each provider are normalized into a shared `ResearchItem` representation while preserving provider-specific metadata.

Examples include:

- GitHub stars and forks
- Repository language and topics
- Hugging Face downloads and likes
- Model pipeline tags
- PapersWithCode tasks
- Conference information
- Publication and update timestamps

---

# Document & Research Workflow

```text
Discover or Upload Source
           ↓
      Normalize Data
           ↓
    Process / Chunk Content
           ↓
     Build Searchable Data
           ↓
   Semantic + Lexical Search
           ↓
     Rank and Filter Results
           ↓
      Select Evidence
           ↓
      Generate Answer
           ↓
      Display Sources
```

arXiv papers can also be processed through the document pipeline when their PDFs are ingested, allowing the same document processing and retrieval mechanisms to be applied to research papers.

---

# LLM Layer

Bujha AI supports separate production and local development inference paths.

## Production

```text
Provider: Groq
Model: openai/gpt-oss-120b
```

## Local Development

```text
Provider: Ollama
Model: qwen3:4b-instruct
```

This separation makes it possible to use local inference during development while using a hosted inference provider for deployment.

---

# Data & Persistence

Bujha AI uses **Supabase / PostgreSQL** for application persistence and **Supabase Storage** for uploaded document files.

The application stores data across entities such as:

```text
workspaces
workspace_sources
workspace_documents
documents
document_chunks
chats
chat_sources
chat_messages
recent_activity
```

### Simplified Storage Architecture

```text
Supabase / PostgreSQL
│
├── Workspaces
├── Sources
├── Documents
├── Document Chunks
├── Chats
├── Messages
└── Recent Activity
     
Supabase Storage
│
└── Uploaded Document Files
```

The document deletion workflow also checks whether an underlying document is referenced by another workspace before deleting its persistent data and stored file.

---

# Chat System

The chat layer provides research-oriented conversations over selected sources.

Features include:

- Persistent chat sessions
- Source attachments
- Document attachments
- Streamed model responses
- Retrieval metadata
- Evidence display
- Source attribution

Chat context can be built from retrieved evidence rather than requiring the user to manually paste research material into the conversation.

---

# Citations

For arXiv-backed research items, Bujha AI can generate:

- APA 7 citations
- BibTeX citations

This makes the system useful for academic research and technical documentation workflows.

---

# Recent Activity

Research activity is persisted at the workspace level.

Tracked activity includes events such as:

- Document added
- Paper added
- Model added
- Repository added
- Chat started
- Research performed

This gives users a persistent view of what has happened inside a workspace.

---

# Technology Stack

## Frontend

- React
- TypeScript
- Vite
- Tailwind-style utility classes
- Lucide React

## Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- Requests
- HTTPX
- python-dotenv

## AI & Retrieval

- Retrieval-Augmented Generation (RAG)
- BM25
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking
- Maximum Marginal Relevance (MMR)
- Query classification
- Query-aware relevance scoring
- NumPy
- scikit-learn

## LLM

- Groq
- Ollama
- Qwen
- `openai/gpt-oss-120b`

## Database & Storage

- Supabase
- PostgreSQL
- Supabase Storage

## Document Processing

- PyPDF
- PyMuPDF
- python-docx
- openpyxl
- python-pptx

## Research APIs

- arXiv
- GitHub REST API
- PapersWithCode
- Hugging Face API
- Hugging Face Datasets Server

## Development

- Git
- GitHub
- pytest
- JSON / filesystem caching

---

# Project Structure

A simplified representation of the project architecture:

```text
.
├── frontend/
│   └── ...
│
├── backend/
│   ├── main.py
│   └── ...
│
├── src/
│   ├── collectors/
│   │   ├── arxiv.py
│   │   ├── github.py
│   │   ├── huggingface.py
│   │   └── paperswithcode.py
│   │
│   ├── clients/
│   │   └── ...
│   │
│   ├── models/
│   │   └── ...
│   │
│   ├── services/
│   │   ├── research.py
│   │   ├── relevance.py
│   │   ├── semantic_search.py
│   │   ├── context_router.py
│   │   ├── github_rag/
│   │   └── document_rag/
│   │
│   └── ...
│
├── tests/
│   ├── test_arxiv.py
│   ├── test_huggingface.py
│   ├── test_paperswithcode.py
│   ├── test_research_service.py
│   └── test_hybrid_retrieval.py
│
└── README.md
```

The backend follows a service-oriented design:

```text
Routes
  ↓
Services
  ↓
Research / RAG Components
  ↓
Persistence / External Providers
```

API routes handle HTTP-level concerns, while services manage application logic, research acquisition, persistence, and retrieval behavior.

Heavy RAG components are initialized lazily to reduce startup memory pressure in constrained deployment environments.

---

# Evaluation

Bujha AI includes a retrieval benchmark to measure retrieval behavior quantitatively.

Current benchmark:

| Metric | Result |
|---|---:|
| Questions | 32 |
| Query Classification Accuracy | 100% |
| MRR | 0.513 |
| Recall@5 | 0.507 |
| Recall@10 | 0.708 |
| nDCG@5 | 0.447 |
| nDCG@10 | 0.532 |

The evaluation includes:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG
- Query classification accuracy
- Per-question failure analysis

The benchmark is intended to identify retrieval strengths and weaknesses rather than to claim universal answer accuracy.

---

# Example Evaluation Workflow

```text
Benchmark Questions
        ↓
Query Classification
        ↓
Hybrid Retrieval
        ↓
Ranking / Reranking
        ↓
Evidence Selection
        ↓
Compare Returned Evidence
        ↓
Calculate:
    Recall@K
    Precision@K
    MRR
    nDCG
        ↓
Failure Analysis
```

This makes retrieval changes measurable and provides a basis for future retrieval experiments.

---

# Design Decisions

## Why Hybrid Retrieval?

Technical research frequently contains both semantic concepts and exact technical identifiers.

For example:

```text
"attention mechanism used by the model"
```

and:

```text
"CrossEntropyLoss"
```

represent different retrieval challenges.

Semantic retrieval helps with conceptual similarity, while BM25 is valuable for exact lexical matching.

Combining the two provides a more balanced retrieval strategy.

---

## Why Structure-Aware Chunking?

The meaning of technical content often depends on its surrounding structure.

For example:

```text
Paper
 └── Methodology
      └── Training Procedure
```

or:

```text
Repository
 └── src/
      └── services/
           └── retriever.py
```

Preserving this structure provides additional context for retrieval and ranking.

---

## Why MMR?

A retrieval result containing five nearly identical chunks provides less useful evidence than several complementary pieces of relevant information.

MMR helps reduce this redundancy while protecting highly relevant evidence.

---

## Why an Embedding Provider Abstraction?

Embedding inference may be performed:

- Locally during development
- Remotely during deployment

Keeping the embedding provider behind an abstraction makes the retrieval layer less dependent on a specific inference environment.

---

## Why Lazy RAG Initialization?

Embedding and retrieval components can be memory-intensive.

Lazy initialization helps reduce the application's startup memory requirements and is useful when deploying the backend on resource-constrained infrastructure.

---

# Current Limitations

Bujha AI is an evolving engineering and experimentation project.

Current limitations include:

- Retrieval quality is not uniform across every query type.
- Some broad overview and methodology queries have lower recall.
- Some precise factual queries remain difficult.
- Dense vector persistence is not currently implemented using `pgvector`.
- The current benchmark contains 32 questions and is therefore relatively small.
- The evaluation should not be interpreted as evidence of large-scale production traffic or commercial-scale usage.

These limitations are intentionally documented because retrieval systems should be evaluated using measurable evidence rather than only qualitative examples.

---

# Future Improvements

Potential directions include:

### Persistent Vector Search

Move persistent dense vectors into PostgreSQL using `pgvector` while retaining the existing retrieval and evidence-selection pipeline.

### Improved Context Reconstruction

Use parent and neighboring chunks to improve context for broad overview and methodology questions.

### Retrieval Ablation Studies

Compare progressively more sophisticated retrieval configurations:

```text
BM25
   ↓
Dense Retrieval
   ↓
Hybrid Retrieval + RRF
   ↓
Hybrid + Reranking
   ↓
Hybrid + Reranking + MMR
```

using consistent retrieval metrics.

### Source Summarization

Introduce a workflow for summarizing selected sources using retrieved evidence instead of summarizing entire datasets indiscriminately.

---

# Installation

## Prerequisites

- Python 3.x
- Node.js
- npm
- Supabase project
- LLM provider credentials

Optional depending on configuration:

- Groq API key
- Ollama
- Hugging Face token
- GitHub token

---

## Clone

```bash
git clone https://github.com/madaneeyy/smart-research-dashboard.git

cd smart-research-dashboard
```

---

## Backend Setup

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the backend environment.

Example:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct

HF_TOKEN=your_huggingface_token
GITHUB_TOKEN=your_github_token
```

Never commit your actual credentials, API keys, or tokens to GitHub.

---

# Run the Backend

```bash
uvicorn backend.main:app --reload
```

---

# Run the Frontend

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

---

# Run Tests

```bash
pytest
```

---

# Deployment

The current architecture supports:

```text
Frontend → Vercel
Backend  → Render
Database → Supabase / PostgreSQL
Storage  → Supabase Storage
LLM      → Groq / Ollama
```

The deployment setup also supports remote embedding inference so the application does not necessarily need to load the local embedding model into the cloud application process.

---

# Security Notes

Before deploying your own instance:

- Keep API keys in environment variables.
- Do not expose private Supabase credentials.
- Do not commit `.env` files.
- Validate uploaded files on the backend.
- Apply appropriate database access policies.
- Restrict external API credentials to the permissions they require.

---

# Why I Built This

Technical research often involves switching between:

- Research papers
- GitHub repositories
- Model repositories
- Documentation
- PDFs
- Spreadsheets
- Presentations
- Multiple conversations

Bujha AI was built to bring these sources together into one research environment and experiment with better ways of retrieving technical evidence before sending context to an LLM.

The project also serves as a practical exploration of **information retrieval, RAG architecture, ranking, document processing, and AI-assisted research workflows**.

---

# What This Project Demonstrates

Bujha AI brings together work across:

- Full-stack application development
- Backend API engineering
- AI integration
- Retrieval-Augmented Generation
- Information retrieval
- Hybrid search
- Semantic search
- BM25
- Ranking and reranking
- Document processing
- Code retrieval
- Research-source integration
- Database design
- Cloud deployment
- Retrieval evaluation

---

# Project Links

**Live Application:**  
https://bujha.vercel.app/

**GitHub Repository:**  
https://github.com/madaneeyy/smart-research-dashboard

---

# Author

## Madan Pandey

Computer Science graduate interested in:

- Artificial Intelligence
- Machine Learning
- Information Retrieval
- RAG Systems
- Software Engineering
- Intelligent Applications
- Research Tools

**GitHub:** https://github.com/madaneeyy  
**LinkedIn:** https://www.linkedin.com/in/madaneeyy/
