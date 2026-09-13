# Bujha AI

### Smart Research Dashboard

Bujha AI is a full-stack AI research workspace for discovering, organizing, retrieving, and querying technical knowledge across research papers, GitHub repositories, machine-learning models, and uploaded documents.

The system combines multi-source research discovery with a hybrid retrieval pipeline consisting of semantic search, BM25, Reciprocal Rank Fusion, relevance filtering, neural reranking, and MMR-based evidence selection before generating grounded responses.

## Features

### Research Discovery

Search and explore technical resources from:

- [arXiv](https://arxiv.org/) — research papers and metadata
- [GitHub](https://github.com/) — repositories and source code
- [Papers With Code](https://paperswithcode.com/) — papers, methods, datasets, and paper-code relationships
- [Hugging Face](https://huggingface.co/) — models and ML resources

Results from different providers are normalized into a common research representation while preserving provider-specific metadata.

### Workspace-Based Research

Research is organized into persistent workspaces containing:

- Research sources
- Uploaded documents
- Papers
- GitHub repositories
- Models
- Chats
- Recent activity

This allows users to keep related research and conversations together instead of managing isolated searches.

### Research Chat

Users can ask questions against selected sources and documents.

The chat layer supports:

- Persistent conversations
- Source and document attachments
- Streaming responses
- Retrieval metadata
- Evidence display
- Source attribution

### Document Processing

Supported formats:

- PDF
- DOCX
- XLSX
- PPTX

Documents are processed into retrieval-friendly chunks while preserving structural metadata such as headings, section hierarchy, page information, chunk type, file path, and chunk indices.

### GitHub Repository Retrieval

Repositories can be used as research sources without sending an entire repository directly to the LLM.

Repository content is:

1. Acquired
2. Filtered
3. Chunked
4. Made searchable
5. Ranked
6. Reduced to relevant evidence

Caching and fallback acquisition paths are used to reduce unnecessary repeated repository retrieval.

### Citations

For arXiv-backed research items, Bujha AI can generate:

- APA 7
- BibTeX

### Recent Activity

Workspace activity is persisted for actions such as:

- Documents added
- Papers added
- Models added
- Repositories added
- Chats started
- Research performed

---

# Architecture

```text
┌───────────────────────────────┐
│       React + TypeScript      │
│           Frontend            │
└───────────────┬───────────────┘
                │
                │ HTTP / JSON / Streaming
                ▼
┌───────────────────────────────┐
│         FastAPI Backend       │
└───────────────┬───────────────┘
                │
        ┌───────┼────────┬──────────────┐
        │       │        │              │
        ▼       ▼        ▼              ▼
   Workspace Research   Chat        Activity
      APIs      APIs    APIs           APIs
        │       │        │              │
        └───────┴────────┴──────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │      RAG Layer      │
             └──────────┬──────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    Semantic          BM25         Query Routing
    Retrieval       Retrieval       & Relevance
         │              │
         └───────┬──────┘
                 ▼
        Reciprocal Rank Fusion
                 │
                 ▼
        Relevance Filtering
                 │
                 ▼
         CrossEncoder Reranking
                 │
                 ▼
          MMR Evidence Selection
                 │
                 ▼
          Grounded LLM Response
```

---

# Retrieval Pipeline

The retrieval pipeline is designed to combine the strengths of lexical and semantic search.

```text
Query
  │
  ▼
Classification / Routing
  │
  ├───────────────┐
  ▼               ▼
Semantic          BM25
Retrieval        Retrieval
  │               │
  └───────┬───────┘
          ▼
     RRF Fusion
          ▼
  Relevance Filtering
          ▼
 CrossEncoder Reranking
          ▼
  MMR Evidence Selection
          ▼
   Grounded Context
          ▼
        LLM
          ▼
     Final Response
```

## Query Classification

Queries are routed into research-oriented categories such as:

- Factual
- Methodology
- Overview
- Comparison
- Limitation
- Cross-source / relationship queries

## Semantic Retrieval

Dense retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding layer supports both local Sentence Transformer inference and remote Hugging Face inference.

## BM25 Retrieval

BM25 provides lexical retrieval for exact technical terminology, identifiers, filenames, symbols, repository-specific terms, and other keyword-sensitive queries.

Searchable metadata can include:

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

## Reciprocal Rank Fusion

Semantic and BM25 rankings are combined using Reciprocal Rank Fusion (RRF).

This provides a single candidate ranking that benefits from both semantic similarity and exact lexical matching.

## Relevance Filtering

Retrieved candidates are filtered using relevance scoring and duplicate removal before final evidence selection.

## CrossEncoder Reranking

For the document RAG path, candidate chunks can undergo second-stage CrossEncoder reranking to improve query-aware ordering.

## MMR Evidence Selection

Maximum Marginal Relevance is used to reduce redundant evidence while retaining highly relevant primary results.

The selection priority is:

```text
Relevance
   ↓
Exact technical matching
   ↓
Complementary evidence
   ↓
Diversity
```

---

# Structure-Aware Chunking

Bujha AI does not rely solely on naive fixed-size text splitting.

The chunking pipeline preserves useful structure including:

- Headings
- Heading hierarchy
- Parent sections
- Section paths
- Paragraphs
- Lists
- Fenced code blocks
- File paths
- Programming language
- Chunk type
- Chunk indices
- Page information

This metadata is available to downstream retrieval and relevance components.

For code repositories, preserving file paths, language, symbols, and structural information helps distinguish otherwise similar chunks.

---

# Research Sources

| Source | Purpose |
|---|---|
| arXiv | Research paper discovery and metadata |
| GitHub | Repository discovery and source-code retrieval |
| Papers With Code | Papers, methods, datasets, and paper-code relationships |
| Hugging Face | Model and ML resource discovery |

External results are normalized into a shared `ResearchItem` representation so that different providers can be handled consistently by the application.

---

# Data Architecture

Bujha AI uses **Supabase / PostgreSQL** for application persistence and **Supabase Storage** for uploaded document files.

Core persistent entities include:

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

### Storage

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
└── Uploaded Documents
```

The document deletion workflow checks whether an underlying document is still referenced by another workspace before removing persistent document data and its stored file.

---

# LLM Providers

Bujha AI supports separate inference paths for production and local development.

### Production

```text
Provider: Groq
Model: openai/gpt-oss-120b
```

### Local Development

```text
Provider: Ollama
Model: qwen3:4b-instruct
```

This keeps the application architecture independent from a single inference environment.

---

# Technology Stack

### Frontend

- React
- TypeScript
- Vite
- Tailwind-style utility classes
- Lucide React

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- Requests
- HTTPX
- python-dotenv

### Retrieval / AI

- Retrieval-Augmented Generation (RAG)
- BM25
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Reciprocal Rank Fusion
- CrossEncoder reranking
- Maximum Marginal Relevance
- Query classification
- Query-aware relevance scoring
- NumPy
- scikit-learn

### Database / Storage

- Supabase
- PostgreSQL
- Supabase Storage

### Document Processing

- PyPDF
- PyMuPDF
- python-docx
- openpyxl
- python-pptx

### Research Integrations

- arXiv
- GitHub REST API
- Papers With Code
- Hugging Face API
- Hugging Face Datasets Server

### Development

- Git
- GitHub
- pytest
- JSON / filesystem caching

---

# Project Structure

A simplified view of the application structure:

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
├── requirements.txt
└── README.md
```

The backend follows a service-oriented architecture:

```text
API Routes
    ↓
Application Services
    ↓
Research / Retrieval Components
    ↓
Persistence + External Providers
```

Heavy retrieval components are initialized lazily to reduce startup memory pressure in constrained deployment environments.

---

# Evaluation

Bujha AI includes a retrieval benchmark to measure search quality quantitatively.

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

These measurements are intended to evaluate the retrieval pipeline and identify weaknesses rather than claim universal answer accuracy.

---

# Development Setup

## Prerequisites

- Python 3.x
- Node.js
- npm
- Supabase project
- LLM provider credentials

Depending on the selected configuration, you may also need:

- Groq API key
- Ollama
- Hugging Face token
- GitHub token

---

## Clone the Repository

```bash
git clone https://github.com/madaneeyy/smart-research-dashboard.git

cd smart-research-dashboard
```

---

## Backend

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

## Environment Variables

Create a `.env` file using your own credentials.

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

Never commit secrets or API keys to the repository.

---

## Run the Backend

```bash
uvicorn backend.main:app --reload
```

---

## Run the Frontend

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

---

## Run Tests

```bash
pytest
```

---

# Deployment

The application is designed around the following deployment architecture:

```text
Frontend
   │
   ▼
Vercel

Backend
   │
   ▼
Render

Database
   │
   ▼
Supabase / PostgreSQL

Document Storage
   │
   ▼
Supabase Storage

LLM
   │
   ├── Groq
   └── Ollama
```

The embedding provider abstraction also supports remote inference so that deployment environments do not necessarily need to load the local embedding model into the application process.

---

# Current Limitations

Bujha AI is an evolving engineering and experimentation project.

Current limitations include:

- Retrieval quality is not uniform across all query types.
- Some overview and methodology queries have lower recall.
- Some precise factual queries remain difficult.
- Dense vector persistence is not currently implemented using `pgvector`.
- The current benchmark contains 32 questions and is relatively small.
- Evaluation results should not be interpreted as evidence of large-scale production traffic or commercial-scale usage.

These limitations are documented intentionally so that system capabilities are represented by measurable evidence rather than by broad claims.

---

# Roadmap

The following areas are candidates for future development:

### Persistent Vector Search

Evaluate PostgreSQL with `pgvector` for persistent dense-vector storage and scalable vector retrieval.

### Improved Context Reconstruction

Use parent and neighboring chunks to improve context for broad overview and methodology queries.

### Retrieval Ablation Studies

Compare retrieval configurations using the same benchmark:

```text
BM25
Dense Retrieval
Hybrid + RRF
Hybrid + Reranking
Hybrid + Reranking + MMR
```

Measure each configuration using:

- Recall@5
- Recall@10
- MRR
- nDCG

### Source Summarization

Add a workflow for summarizing selected sources using the existing retrieval and evidence pipeline.

---

# Engineering Principles

Bujha AI follows a few core design principles:

**Evidence before generation**  
Retrieve relevant evidence before asking the LLM to generate an answer.

**Hybrid retrieval over single-method search**  
Use both lexical and semantic retrieval because technical queries often require both.

**Structure-aware processing**  
Preserve document and repository structure where it contributes to retrieval quality.

**Measured improvements**  
Evaluate retrieval changes with benchmark metrics instead of relying only on qualitative examples.

**Deployment-aware architecture**  
Keep inference and retrieval components modular so the application can operate under constrained infrastructure.

---

# What the Project Demonstrates

Bujha AI brings together practical work across:

- Full-stack application development
- Backend API engineering
- AI integration
- Retrieval-Augmented Generation
- Information retrieval
- Semantic search
- BM25
- Hybrid retrieval
- Ranking and reranking
- Document processing
- Code retrieval
- Research-source integration
- Database design
- Cloud deployment
- Retrieval evaluation

---

# Repository

**GitHub:**  
https://github.com/madaneeyy/smart-research-dashboard

---

# Author

**Madan Pandey**

Computer Science graduate focused on AI/ML, information retrieval, software engineering, and intelligent applications.

- GitHub: https://github.com/madaneeyy
- LinkedIn: https://www.linkedin.com/in/madaneeyy/

---

## License

This project is currently maintained as a personal engineering and research project.


