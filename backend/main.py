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

    # --------------------------------------------------------
    # Build system prompt
    # --------------------------------------------------------

    system_prompt = f"""
You are an AI research assistant for a research
discovery application.

Your job is to help the user understand the research
resource provided by the application.

IMPORTANT RULES:

1. Answer the user's question using the research
   information provided below.

2. Do not ask the user to provide information that is
   already present in the research context.

3. Do not invent information that is not present in
   the research context.

4. If the available information is insufficient to
   answer the question, clearly say that the available
   research information is insufficient.

5. You may summarize, explain, organize, and interpret
   information that is explicitly present in the context.

6. Clearly distinguish between facts explicitly stated
   in the research information and reasonable interpretations.

7. Do not pretend that you have read the complete paper
   or repository if the supplied context does not contain
   that information.

8. Keep answers clear and reasonably concise.

9. When the context contains technical information,
   explain it in simple language when appropriate.

10. Focus on answering the user's actual question rather
    than giving unrelated information.

The research information supplied by the application is:

============================================================
RESEARCH CONTEXT
============================================================

{context}

============================================================
END RESEARCH CONTEXT
============================================================
"""

    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    user_prompt = f"""
Based strictly on the research information provided above,
answer the following question:

{question}
"""

    # --------------------------------------------------------
    # Send request to Ollama
    # --------------------------------------------------------

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
                # Keep the context reasonably sized for
                # your laptop's available memory.
                "num_ctx": 8192,

                # Keep answers reasonably concise.
                "temperature": 0.2,
                "num_predict": 450,
            },

            # Keep the model loaded for a few minutes after
            # the request so repeated questions don't require
            # a complete reload every time.
            keep_alive="5m",
        )

        answer = response.message.content

        if not answer:

            raise ValueError(
                "Ollama returned an empty response."
            )

        return {
            "question": question,
            "answer": answer,
            "model": OLLAMA_MODEL,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Ollama request failed: "
                f"{str(e)}"
            ),
        )