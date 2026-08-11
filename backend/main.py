from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama


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
        "message": "Smart Research AI API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        response = ollama.list()

        return {
            "status": "ok",
            "ollama": "connected",
            "models": [
                model.model
                for model in response.models
            ],
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ollama connection failed: {str(e)}",
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
    # Build AI prompt
    # --------------------------------------------------------

    system_prompt = """
You are an AI research assistant for a research discovery
application.

Your job is to help the user understand the research
resource provided by the application.

IMPORTANT RULES:

1. Answer the user's question using the research information
   provided below.

2. Do not ask the user to provide the title, description,
   URL, or other information that is already present in the
   research context.

3. Do not invent information that is not present in the
   research context.

4. If the available information is insufficient to answer
   the question, clearly say that the available research
   metadata is insufficient.

5. You may explain or summarize information that is explicitly
   present in the context.

6. Keep the answer clear, useful, and reasonably concise.

7. When discussing the research, distinguish between facts
   explicitly present in the metadata and reasonable
   interpretations.

The research information supplied by the application is:

----------------------------------------
RESEARCH CONTEXT
----------------------------------------

""" + context

    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    user_prompt = f"""
Based on the research information provided above, answer
the following question:

{question}
"""

    # --------------------------------------------------------
    # Send request to Ollama
    # --------------------------------------------------------

    try:

        response = ollama.chat(
            model="qwen3:8b",
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
        )

        answer = response.message.content

        return {
            "question": question,
            "answer": answer,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ollama request failed: {str(e)}",
        )