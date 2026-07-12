"""Phase 3 — Cloud AI 100 interface: builds a grounded prompt from the
query + reranked chunks and runs LLM generation on a Cloud AI 100-hosted
model, via Cirrascale's Imagine SDK (aisuite.cirrascale.com), which is the
supported way to reach a Cloud AI 100 endpoint over HTTP without needing
the low-level, on-device Qualcomm AI SDK.

Wiring: CloudAI100Client is configured entirely from environment
variables, so no code changes are needed to go live — see
SNAPDRAGON_PC_SETUP.md for the full walkthrough:
    IMAGINE_API_KEY       required. Your Imagine SDK API key.
    IMAGINE_ENDPOINT_URL  optional. Defaults to whatever ImagineClient()
                          itself defaults to when unset.
    IMAGINE_MODEL_NAME    optional. The Cloud AI 100-hosted model name to
                          call (see IMAGINE_API_KEY's account for which
                          models are available). Defaults to "Llama-3.1-8B".

If IMAGINE_API_KEY isn't set, or the `imagine` package (Imagine SDK 0.4.2)
isn't installed, CloudAI100Client raises NotImplementedError at
construction time — the same signal cloud_client.rerank_and_generate()
already treats as "not configured yet" and falls back from, so /query
keeps working with a templated answer until real credentials are provided.

Input: query text + reranked chunk rows (from cloud.reranker.rerank()), or
just query text when nothing was found to ground the answer in.
Output: build_prompt()/build_general_prompt() -> str;
generate_answer()/generate_general_answer() -> str (the model's answer).
Side effects: CloudAI100Client.__init__() reads env vars and constructs an
Imagine SDK client (no network call yet); CloudAI100Client.generate()
makes a real HTTP call to the Cloud AI 100-hosted model.
"""

import logging
import os

logger = logging.getLogger("lore")

MAX_CHUNK_CHARS = 500

DEFAULT_MODEL_NAME = "Llama-3.1-8B"


def build_prompt(query, reranked_chunks):
    """Build a grounded prompt instructing the model to answer only from
    the given chunks and cite sources by title.

    Args:
        query: the user's original query text.
        reranked_chunks: chunk row dicts (title, chunk, location, ...),
            nearest-first per cloud.reranker.rerank().

    Returns:
        A prompt string listing each source (numbered, titled, excerpted)
        followed by instructions to answer only from those sources and
        cite them by title.
    """
    source_lines = []
    for index, chunk in enumerate(reranked_chunks, start=1):
        title = chunk.get("title") or "untitled source"
        text = (chunk.get("chunk") or "").strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rstrip() + "..."
        source_lines.append(f"[{index}] {title}: {text}")
    sources_block = "\n".join(source_lines) if source_lines else "(no sources provided)"

    return (
    "You are Lore, a private, local-first AI knowledge assistant designed to "
    "search, reason over, and synthesize information from a user's personal "
    "knowledge base. Your purpose is to help users retrieve information from "
    "their own documents accurately, privately, and transparently.\n\n"

    "=============================\n"
    "IDENTITY\n"
    "=============================\n"
    "You are NOT a general-purpose chatbot.\n"
    "You are a retrieval-augmented AI assistant whose knowledge for this "
    "conversation comes ONLY from the retrieved documents provided below.\n\n"

    "Every response must be grounded in those retrieved sources.\n"
    "Do not rely on your pretraining or external knowledge, even if you believe "
    "you know the answer.\n\n"

    "=============================\n"
    "PRIMARY OBJECTIVE\n"
    "=============================\n"
    "Your objective is to provide the most accurate, truthful, and well-supported "
    "answer possible using ONLY the retrieved sources.\n\n"

    "Your highest priorities are:\n"
    "1. Accuracy\n"
    "2. Faithfulness to the retrieved documents\n"
    "3. Correct source attribution\n"
    "4. Clear reasoning\n"
    "5. Concise but complete explanations\n\n"

    "=============================\n"
    "GROUNDING RULES\n"
    "=============================\n"
    "- Treat the retrieved documents as the ONLY source of truth.\n"
    "- Never fabricate information.\n"
    "- Never invent facts.\n"
    "- Never guess missing information.\n"
    "- Never assume details that are not explicitly supported.\n"
    "- Never answer from world knowledge.\n"
    "- If something is not supported by the retrieved documents, explicitly "
    "state that the information is unavailable.\n"
    "- If the sources only partially answer the question, clearly explain what "
    "is known and what remains unknown.\n\n"

    "=============================\n"
    "REASONING PROCESS\n"
    "=============================\n"
    "Before generating your final answer, internally:\n"
    "1. Understand the user's intent.\n"
    "2. Identify which retrieved sources are relevant.\n"
    "3. Compare information across multiple sources.\n"
    "4. Resolve contradictions when possible.\n"
    "5. Synthesize information into a coherent answer.\n"
    "6. Ensure every factual claim is supported by at least one retrieved source.\n\n"

    "Do not reveal this reasoning process.\n"
    "Only output the final answer.\n\n"

    "=============================\n"
    "SOURCE USAGE\n"
    "=============================\n"
    "- Use as many relevant sources as necessary.\n"
    "- Prefer combining multiple relevant sources instead of relying on a single "
    "document.\n"
    "- Avoid repeating identical information.\n"
    "- If two sources provide complementary information, merge them naturally.\n"
    "- If two sources disagree, explain the disagreement instead of choosing one "
    "without justification.\n\n"

    "=============================\n"
    "CITATIONS\n"
    "=============================\n"
    "- Every factual statement should be supported by citations.\n"
    "- Cite the title of every source you use in square brackets.\n"
    "- When multiple sources support a statement, cite all of them.\n"
    "- Do not cite sources that were not used.\n\n"

    "Example:\n"
    "The embedding model generates semantic vectors for document retrieval "
    "[Embedding Pipeline.md][Vector Database Design.pdf]\n\n"

    "=============================\n"
    "WHEN INFORMATION IS MISSING\n"
    "=============================\n"
    "If the retrieved sources do not contain enough information:\n"
    "- Clearly say so.\n"
    "- Explain exactly what information is missing.\n"
    "- Do not speculate.\n"
    "- Do not fill gaps using prior knowledge.\n\n"

    "=============================\n"
    "HANDLING AMBIGUITY\n"
    "=============================\n"
    "If the user's question is ambiguous:\n"
    "- Infer the most likely interpretation using the retrieved sources.\n"
    "- If multiple interpretations are plausible, briefly explain them.\n"
    "- Do not invent details.\n\n"

    "=============================\n"
    "STYLE\n"
    "=============================\n"
    "- Be professional.\n"
    "- Be concise.\n"
    "- Be technically accurate.\n"
    "- Avoid unnecessary verbosity.\n"
    "- Prefer structured explanations.\n"
    "- Use bullet points where appropriate.\n"
    "- Preserve important technical terminology.\n\n"

    "=============================\n"
    "OUTPUT FORMAT\n"
    "=============================\n"
    "Whenever appropriate, structure responses as:\n"
    "- Direct answer\n"
    "- Supporting explanation\n"
    "- Key observations\n"
    "- Citations\n\n"

    "If the answer is short, simply provide a concise response with citations.\n\n"

    "=============================\n"
    "RETRIEVED SOURCES\n"
    "=============================\n"
    f"{sources_block}\n\n"

    "=============================\n"
    "USER QUESTION\n"
    "=============================\n"
    f"{query}\n\n"

    "=============================\n"
    "FINAL ANSWER\n"
    "=============================\n"
)


def build_general_prompt(query):
    """Build a prompt for when no indexed sources matched the query at all
    (nothing's been indexed yet, or nothing relevant exists) — unlike
    build_prompt(), this doesn't restrict the model to citing sources; it
    draws on general knowledge instead. Styled to read like a factual
    article excerpt rather than a chatbot reply, so a fallback answer
    doesn't look jarringly different from a grounded one — the UI already
    discloses "0 sources found" separately, so the answer text itself
    doesn't need a "based on my knowledge" disclaimer cluttering it.

    Args:
        query: the user's original query text.

    Returns:
        A prompt string instructing the model to answer generally, in an
        article-like voice.
    """
    return (
        "You are Lore, a private on-device assistant. No documents in the "
        "user's personal index matched this question, so answer using "
        "your own general knowledge instead. Write the answer the way a "
        "neutral, factual reference article would state it: direct, "
        "third-person prose that states the facts plainly, as if it were "
        "an excerpt from an article on the subject. Do not use "
        "conversational framing or disclaimers like 'as an AI' or 'based "
        "on my knowledge' — just answer.\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


class CloudAI100Client:
    """Cloud AI 100 interface via Cirrascale's Imagine SDK (0.4.2).

    Configuration is env-var driven (see module docstring) so that
    providing IMAGINE_API_KEY is the only step needed to go from the
    templated fallback to real Cloud AI 100-generated answers.
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("IMAGINE_MODEL_NAME") or DEFAULT_MODEL_NAME
        self._chat_message_cls = None
        self._session = self._load_session()

    def _load_session(self):
        api_key = os.environ.get("IMAGINE_API_KEY")
        if not api_key:
            raise NotImplementedError(
                "CloudAI100Client has no IMAGINE_API_KEY set, so there's "
                "nothing to connect to yet. Set the IMAGINE_API_KEY "
                "environment variable (and optionally IMAGINE_ENDPOINT_URL "
                "/ IMAGINE_MODEL_NAME) to enable real Cloud AI 100 "
                "inference via the Imagine SDK — see SNAPDRAGON_PC_SETUP.md."
            )

        try:
            import imagine as imagine_sdk
        except ImportError as exc:
            raise NotImplementedError(
                "The 'imagine' package (Imagine SDK 0.4.2, Cirrascale AI "
                "Suite) isn't installed. Install the wheel per "
                "https://aisuite.cirrascale.com/sdk/install.html, then retry."
            ) from exc

        self._chat_message_cls = imagine_sdk.ChatMessage
        endpoint = os.environ.get("IMAGINE_ENDPOINT_URL")
        logger.info(
            "Cloud AI 100 client configured via Imagine SDK (model=%s, endpoint=%s)",
            self.model_name,
            endpoint or "<sdk default>",
        )
        return imagine_sdk.ImagineClient(api_key=api_key, endpoint=endpoint)

    def generate(self, prompt):
        message = self._chat_message_cls(role="user", content=prompt)
        response = self._session.chat(messages=[message], model=self.model_name)
        return response.first_content


def generate_answer(query, reranked_chunks, client):
    """Generate a grounded answer for `query` using `reranked_chunks` as
    context, via `client.generate()`.

    Args:
        query: the user's original query text.
        reranked_chunks: chunk row dicts, nearest-first.
        client: any object exposing generate(prompt: str) -> str (a real
            CloudAI100Client, or a test double).

    Returns:
        The model's answer text (client.generate()'s return value).
    Side effects: whatever client.generate() does (a real HTTP call to the
        Cloud AI 100-hosted model, once CloudAI100Client is configured).
    """
    prompt = build_prompt(query, reranked_chunks)
    return client.generate(prompt)


def generate_general_answer(query, client):
    """Generate an answer with no grounding sources at all — the
    no-candidates-found fallback path. See build_general_prompt().

    Args:
        query: the user's original query text.
        client: any object exposing generate(prompt: str) -> str (a real
            CloudAI100Client, or a test double).

    Returns:
        The model's answer text (client.generate()'s return value).
    Side effects: whatever client.generate() does (a real HTTP call to the
        Cloud AI 100-hosted model, once CloudAI100Client is configured).
    """
    return client.generate(build_general_prompt(query))
