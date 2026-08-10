from dotenv import load_dotenv
from openai import OpenAI

from src.models.research import ResearchItem


load_dotenv()
class AIResearchService:
    """
    AI-powered research assistant.

    Phase 1:
    Uses the metadata already available in a ResearchItem.

    It does NOT download or analyze the full paper yet.
    """

    def __init__(self):
        self.client = OpenAI()

    # =========================================================
    # BUILD RESEARCH CONTEXT
    # =========================================================

    @staticmethod
    def _build_context(
        item: ResearchItem,
    ) -> str:
        """
        Convert a ResearchItem into structured context
        that can be provided to the AI.
        """

        context_parts = []

        # -----------------------------------------------------
        # Basic information
        # -----------------------------------------------------

        context_parts.append(
            f"Title: {item.title}"
        )

        context_parts.append(
            f"Source: {item.source}"
        )

        context_parts.append(
            f"URL: {item.url}"
        )

        # -----------------------------------------------------
        # Description
        # -----------------------------------------------------

        if item.description:
            context_parts.append(
                f"Description: {item.description}"
            )

        # -----------------------------------------------------
        # Authors
        # -----------------------------------------------------

        if item.authors:
            context_parts.append(
                "Authors: "
                + ", ".join(item.authors)
            )

        # -----------------------------------------------------
        # Tags
        # -----------------------------------------------------

        if item.tags:
            context_parts.append(
                "Tags: "
                + ", ".join(item.tags)
            )

        # -----------------------------------------------------
        # Dates
        # -----------------------------------------------------

        if item.published:
            context_parts.append(
                f"Published: {item.published}"
            )

        if item.updated:
            context_parts.append(
                f"Updated: {item.updated}"
            )

        # -----------------------------------------------------
        # GitHub metadata
        # -----------------------------------------------------

        if item.stars is not None:
            context_parts.append(
                f"GitHub stars: {item.stars}"
            )

        if item.forks is not None:
            context_parts.append(
                f"GitHub forks: {item.forks}"
            )

        if item.language:
            context_parts.append(
                f"Programming language: {item.language}"
            )

        # -----------------------------------------------------
        # Hugging Face metadata
        # -----------------------------------------------------

        if item.downloads is not None:
            context_parts.append(
                f"Downloads: {item.downloads}"
            )

        if item.likes is not None:
            context_parts.append(
                f"Likes: {item.likes}"
            )

        if item.library:
            context_parts.append(
                f"Library: {item.library}"
            )

        if item.pipeline_tag:
            context_parts.append(
                f"Pipeline tag: {item.pipeline_tag}"
            )

        # -----------------------------------------------------
        # PapersWithCode metadata
        # -----------------------------------------------------

        if item.tasks:
            context_parts.append(
                "Tasks: "
                + ", ".join(item.tasks)
            )

        if item.conference:
            context_parts.append(
                f"Conference: {item.conference}"
            )

        return "\n".join(context_parts)

    # =========================================================
    # ASK AI
    # =========================================================

    def ask(
        self,
        item: ResearchItem,
        question: str,
    ) -> str:
        """
        Ask the AI a question about a research item.

        The AI is restricted to the information available
        in the ResearchItem.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "question must not be empty"
            )

        # -----------------------------------------------------
        # Build research context
        # -----------------------------------------------------

        research_context = (
            self._build_context(item)
        )

        # -----------------------------------------------------
        # System prompt
        # -----------------------------------------------------

        system_prompt = """
You are an AI research assistant.

Your job is to help users understand research papers,
repositories, models, and other research resources.

You are currently operating in Phase 1.

You only have access to the metadata supplied by the
application. You do NOT have access to the full paper,
full repository, PDF, or external sources.

IMPORTANT RULES:

1. Only make claims supported by the provided research
   information.

2. Do not invent methodology, datasets, experiments,
   results, architectures, or conclusions that are not
   present in the supplied information.

3. If the available information is insufficient to answer
   the question, explicitly say that the available metadata
   is insufficient.

4. Explain technical concepts clearly.

5. When appropriate, distinguish between what is explicitly
   stated in the research information and what can reasonably
   be inferred.

6. Keep answers useful and reasonably concise.

Research information:
"""

        # -----------------------------------------------------
        # User prompt
        # -----------------------------------------------------

        user_prompt = f"""
{research_context}

User question:
{question}
"""

        # -----------------------------------------------------
        # OpenAI request
        # -----------------------------------------------------

        response = self.client.responses.create(
            model="gpt-5-mini",
            instructions=system_prompt,
            input=user_prompt,
        )

        return response.output_text