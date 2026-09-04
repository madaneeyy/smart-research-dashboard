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
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/RAG-GenAI-7C3AED" alt="RAG">
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?logo=supabase&logoColor=white" alt="Supabase">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Problem](#problem)
- [Goals](#goals)
- [Screenshots](#screenshots)
- [Core Capabilities](#core-capabilities)
  - [Research Discovery](#research-discovery)
  - [Document-Based RAG](#document-based-rag)
  - [GitHub Repository Research](#github-repository-research)
  - [LLM-Powered Answers](#llm-powered-answers)
  - [Evidence and Source Attribution](#evidence-and-source-attribution)
- [System Architecture](#system-architecture)
- [Application Flow](#application-flow)
- [RAG Pipeline](#rag-pipeline)
- [Query Classification](#query-classification)
- [Document Acquisition](#document-acquisition)
- [Document Processing and Chunking](#document-processing-and-chunking)
- [Hybrid Retrieval](#hybrid-retrieval)
  - [BM25](#bm25)
  - [TF-IDF](#tf-idf)
  - [Dense Retrieval](#dense-retrieval)
  - [Score Fusion](#score-fusion)
- [Reranking and Evidence Selection](#reranking-and-evidence-selection)
  - [Cross-Encoder Reranking](#cross-encoder-reranking)
  - [Metadata-Aware Ranking](#metadata-aware-ranking)
  - [Query-Fit Scoring](#query-fit-scoring)
  - [Relevance Filtering](#relevance-filtering)
  - [Duplicate and Redundancy Filtering](#duplicate-and-redundancy-filtering)
  - [Maximum Marginal Relevance](#maximum-marginal-relevance)
- [Context Construction](#context-construction)
- [GitHub Repository Retrieval](#github-repository-retrieval)
  - [Repository Metadata](#repository-metadata)
  - [Repository Tree](#repository-tree)
  - [Query-Focused Repository Retrieval](#query-focused-repository-retrieval)
  - [Code-Oriented Questions](#code-oriented-questions)
- [Document Sources](#document-sources)
- [LLM Integration](#llm-integration)
  - [Groq](#groq)
  - [Ollama](#ollama)
  - [LLM Workflow](#llm-workflow)
- [Evaluation](#evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Benchmark Results](#benchmark-results)
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

**Smart Research Dashboard** is a full-stack AI research assistant for discovering, searching, analyzing, and understanding research papers, technical documents, and GitHub repositories.

The project combines traditional information retrieval with semantic search, reranking, evidence selection, and Large Language Models (LLMs) to build a research workflow that is more structured than a simple LLM chat application.

The core idea is:

> **Retrieve relevant evidence first, then use an LLM to generate an answer from that context.**

The system currently combines:

- Research discovery
- Document ingestion
- Document chunking
- Hybrid information retrieval
- BM25 retrieval
- TF-IDF retrieval
- Dense semantic embeddings
- Reciprocal Rank Fusion
- Query classification
- Query-aware retrieval
- Metadata-aware scoring
- Cross-encoder reranking
- Relevance filtering
- Duplicate and near-duplicate filtering
- Maximum Marginal Relevance (MMR)
- Evidence selection
- GitHub repository retrieval
- LLM-based answer generation
- Source and evidence attribution
- Retrieval evaluation

The overall workflow is:

```text
User Question
      |
      v
Query Analysis
      |
      v
Source / Document Acquisition
      |
      v
Document Processing
      |
      v
Candidate Retrieval
      |
      v
Candidate Ranking
      |
      v
Cross-Encoder Reranking
      |
      v
Evidence Selection
      |
      v
Context Construction
      |
      v
LLM
      |
      v
Answer + Supporting Evidence
