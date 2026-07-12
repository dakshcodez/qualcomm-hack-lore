import sys
import types

import pytest

from cloud.inference import (
    CloudAI100Client,
    build_general_prompt,
    build_prompt,
    generate_answer,
    generate_general_answer,
)


class FakeClient:
    def __init__(self, answer="a fake answer"):
        self.answer = answer
        self.received_prompt = None

    def generate(self, prompt):
        self.received_prompt = prompt
        return self.answer


def test_build_prompt_includes_query_and_numbered_sources():
    chunks = [
        {"title": "notes.txt", "chunk": "the hackathon starts July 11"},
        {"title": "paper.pdf", "chunk": "neural embeddings research"},
    ]
    prompt = build_prompt("when does it start", chunks)

    assert "when does it start" in prompt
    assert "[1] notes.txt: the hackathon starts July 11" in prompt
    assert "[2] paper.pdf: neural embeddings research" in prompt
    assert prompt.strip().endswith("Answer:")


def test_build_prompt_instructs_citation_and_grounding():
    prompt = build_prompt("query", [{"title": "a", "chunk": "b"}])
    assert "cite" in prompt.lower()
    assert "only" in prompt.lower()


def test_build_prompt_with_no_chunks_notes_no_sources():
    prompt = build_prompt("query", [])
    assert "(no sources provided)" in prompt
    assert "query" in prompt


def test_build_prompt_truncates_long_chunk_text():
    long_text = "word " * 200  # far more than MAX_CHUNK_CHARS
    prompt = build_prompt("query", [{"title": "long", "chunk": long_text}])
    assert "..." in prompt
    assert len(prompt) < len(long_text) + 500


def test_build_prompt_handles_missing_title():
    prompt = build_prompt("query", [{"chunk": "content with no title"}])
    assert "untitled source" in prompt


def test_build_general_prompt_includes_the_query():
    prompt = build_general_prompt("where is the eiffel tower")
    assert "where is the eiffel tower" in prompt
    assert prompt.strip().endswith("Answer:")


def test_build_general_prompt_does_not_restrict_to_sources():
    # This is the "nothing indexed matched" fallback — it must not carry
    # build_prompt()'s "answer ONLY from the sources" restriction, or the
    # model would just refuse to answer.
    prompt = build_general_prompt("query")
    assert "ONLY the sources below" not in prompt
    assert "general knowledge" in prompt.lower()


def test_build_general_prompt_tells_the_model_to_flag_the_answer_as_unsourced():
    prompt = build_general_prompt("query")
    assert "isn't based on the user's own files" in prompt


def test_generate_general_answer_calls_client_with_the_general_prompt():
    fake = FakeClient(answer="Paris, France.")

    answer = generate_general_answer("where is the eiffel tower", fake)

    assert answer == "Paris, France."
    assert fake.received_prompt == build_general_prompt("where is the eiffel tower")


def test_cloud_ai100_client_raises_not_implemented_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("IMAGINE_API_KEY", raising=False)
    with pytest.raises(NotImplementedError, match="IMAGINE_API_KEY"):
        CloudAI100Client()


def test_cloud_ai100_client_raises_not_implemented_when_imagine_package_missing(monkeypatch):
    monkeypatch.setenv("IMAGINE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "imagine", None)  # simulate `import imagine` failing
    with pytest.raises(NotImplementedError, match="imagine.*package"):
        CloudAI100Client()


def _install_fake_imagine_module(monkeypatch, chat_impl):
    class ChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class FakeResponse:
        def __init__(self, content):
            self.first_content = content

    class ImagineClient:
        def __init__(self, api_key=None, endpoint=None):
            self.api_key = api_key
            self.endpoint = endpoint

        def chat(self, messages, model):
            return FakeResponse(chat_impl(messages, model))

    fake_module = types.SimpleNamespace(ChatMessage=ChatMessage, ImagineClient=ImagineClient)
    monkeypatch.setitem(sys.modules, "imagine", fake_module)


def test_cloud_ai100_client_calls_imagine_sdk_with_configured_model(monkeypatch):
    monkeypatch.setenv("IMAGINE_API_KEY", "test-key")
    monkeypatch.setenv("IMAGINE_MODEL_NAME", "some-cloud-ai-100-model")
    received = {}

    def chat_impl(messages, model):
        received["messages"] = messages
        received["model"] = model
        return "a real answer"

    _install_fake_imagine_module(monkeypatch, chat_impl)

    client = CloudAI100Client()
    answer = client.generate("what is the meaning of life")

    assert answer == "a real answer"
    assert received["model"] == "some-cloud-ai-100-model"
    assert received["messages"][0].role == "user"
    assert received["messages"][0].content == "what is the meaning of life"


def test_cloud_ai100_client_defaults_model_name_when_unset(monkeypatch):
    monkeypatch.setenv("IMAGINE_API_KEY", "test-key")
    monkeypatch.delenv("IMAGINE_MODEL_NAME", raising=False)
    _install_fake_imagine_module(monkeypatch, chat_impl=lambda messages, model: "ok")

    client = CloudAI100Client()

    assert client.model_name  # a non-empty default, not left unset


def test_generate_answer_calls_client_with_built_prompt():
    fake = FakeClient(answer="hackathon starts July 11, per notes.txt")
    chunks = [{"title": "notes.txt", "chunk": "the hackathon starts July 11"}]

    answer = generate_answer("when does it start", chunks, fake)

    assert answer == "hackathon starts July 11, per notes.txt"
    assert fake.received_prompt == build_prompt("when does it start", chunks)


def test_generate_answer_propagates_client_exceptions():
    class FailingClient:
        def generate(self, prompt):
            raise RuntimeError("hardware unavailable")

    with pytest.raises(RuntimeError, match="hardware unavailable"):
        generate_answer("query", [], FailingClient())
