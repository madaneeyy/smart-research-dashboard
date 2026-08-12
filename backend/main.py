import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

try:
    from dotenv import load_dotenv

    load_dotenv()

except ImportError:
    pass


# ============================================================
# RAG IMPORTS
# ============================================================

from src.services.rag.chunker import TextChunker
from src.services.rag.retriever import SimpleRetriever


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)


# Number of chunks sent to the LLM
TOP_K = 2


# ============================================================
# OLLAMA CLIENT
# ============================================================

ollama_client = ollama.Client(
    host=OLLAMA_HOST
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Smart Research AI API",
    description="AI backend for Smart Research Dashboard",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    question: str
    context: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Smart Research AI API is running",
        "model": OLLAMA_MODEL,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        response = ollama_client.list()

        models = [
            model.model
            for model in response.models
        ]

        return {
            "status": "ok",
            "ollama": "connected",
            "model": OLLAMA_MODEL,
            "models": models,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Ollama connection failed: "
                f"{str(e)}"
            ),
        )


# ============================================================
# ASK AI
# ============================================================

@app.post("/ask")
def ask_ai(request: AskRequest):

    question = request.question.strip()
    context = request.context.strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    # --------------------------------------------------------
    # Validate context
    # --------------------------------------------------------

    if not context:

        raise HTTPException(
            status_code=400,
            detail="Research context must not be empty.",
        )

    # ========================================================
    # RAG STEP 1 — CHUNK THE CONTEXT
    # ========================================================

    try:

        chunks = TextChunker.chunk_text(
            context
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to chunk research context: "
                f"{str(e)}"
            ),
        )

    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="No usable research chunks were created.",
        )

    # ========================================================
    # RAG STEP 2 — RETRIEVE RELEVANT CHUNKS
    # ========================================================

    try:

        retrieved_chunks = SimpleRetriever.retrieve(
            question=question,
            chunks=chunks,
            top_k=TOP_K,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve relevant research "
                f"chunks: {str(e)}"
            ),
        )

    if not retrieved_chunks:

        raise HTTPException(
            status_code=400,
            detail="No relevant research information was found.",
        )

    # ========================================================
    # RAG STEP 3 — BUILD RETRIEVED CONTEXT
    # ========================================================

    retrieved_context = "\n\n".join(
        f"--- Research Chunk {i + 1} ---\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    # ========================================================
    # BUILD SYSTEM PROMPT
    # ========================================================

    system_prompt = f"""
You are an AI research assistant for a research
discovery application.

Your job is to help the user understand the research
resource provided by the application.

IMPORTANT RULES:

1. Answer the user's question using ONLY the
   retrieved research information below.

2. Do not invent facts that are not supported
   by the retrieved research information.

3. If the retrieved information is insufficient
   to answer the question, clearly say so.

4. You may summarize, explain, organize, and
   interpret information explicitly present
   in the retrieved context.

5. Clearly distinguish facts from reasonable
   interpretations.

6. Do not pretend that you have read the complete
   repository, paper, or research resource if
   the retrieved context does not contain that
   information.

7. Keep answers clear and reasonably concise.

8. When technical information is present,
   explain it in simple language when appropriate.

9. Focus directly on the user's question.

10. Prefer structured answers when they improve
    clarity.

The following information was retrieved from
the research resource because it is relevant
to the user's question:

============================================================
RETRIEVED RESEARCH CONTEXT
============================================================

{retrieved_context}

============================================================
END RETRIEVED RESEARCH CONTEXT
============================================================
"""

    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
Based strictly on the retrieved research information,
answer the following question:

{question}
"""

    # ========================================================
    # SEND REQUEST TO OLLAMA
    # ========================================================

    try:

        response = ollama_client.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            options={

                # Context window
                "num_ctx": 8192,

                # More deterministic answers
                "temperature": 0.2,

                # Limit response length
                "num_predict": 450,
            },

            # Keep model loaded for repeated questions
            keep_alive="5m",
        )

        answer = response.message.content

        if not answer:

            raise ValueError(
                "Ollama returned an empty response."
            )

        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return {
            "question": question,
            "answer": answer,
            "model": OLLAMA_MODEL,

            # Useful for debugging RAG
            "chunks_retrieved": len(retrieved_chunks),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Ollama request failed: "
                f"{str(e)}"
            ),
        )