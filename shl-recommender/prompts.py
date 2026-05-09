"""
Prompt templates used by the conversational agent.

Design notes:
- Keep prompts centralized to make iteration easy.
- Prefer structured outputs (JSON) for recommendations and tool routing.
- Add explicit constraints to reduce hallucination and ensure URL fidelity.

TODO:
- Add prompt versions and A/B testing hooks.
- Add separate prompts for clarify/recommend/refine/compare behaviors.
- Add safety policies + catalog grounding instructions.
"""

from __future__ import annotations


# System-level instructions for the LLM (high-level behavior).
SYSTEM_PROMPT: str = """\
You are an SHL assessment recommender assistant.
You must ground recommendations in the provided catalog context.
If the user intent is unclear, ask concise clarifying questions before recommending.
When recommending, return a helpful natural-language reply plus a structured list of items.
"""


# Prompt for generating clarifying questions.
CLARIFY_PROMPT: str = """\
Given the user's message, ask the minimum number of clarifying questions needed
to recommend appropriate SHL assessments.

User message:
{user_message}
"""


# Prompt for producing recommendations given user requirements and retrieved catalog context.
RECOMMEND_PROMPT: str = """\
Use the retrieved catalog context to recommend assessments that best match the user's needs.
Only recommend items present in the context.

User requirements:
{requirements}

Retrieved catalog context:
{catalog_context}

Return:
1) A brief helpful reply.
2) A JSON array of recommendations with fields: name, url, test_type.
"""


# Prompt for refining recommendations after user feedback.
REFINE_PROMPT: str = """\
Refine the previous recommendations based on the user's feedback.
Only recommend items present in the retrieved catalog context.

Previous recommendations (JSON):
{previous_recommendations}

User feedback:
{user_feedback}

Retrieved catalog context:
{catalog_context}
"""


# Prompt for comparing a small set of candidate assessments.
COMPARE_PROMPT: str = """\
Compare the following assessments for the user's scenario.
Focus on practical trade-offs and when to choose each.

User scenario:
{scenario}

Candidates (JSON):
{candidates}
"""

