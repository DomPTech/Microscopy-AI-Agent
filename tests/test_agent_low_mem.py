import os

from unittest import mock

from app.agent import core


def test_agent_fallback_to_small_model(monkeypatch):
    # Force low memory path
    monkeypatch.setattr(core, "_is_low_memory", lambda threshold_gb=12: True)

    class DummyModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        # minimal interface required by CodeAgent in tests
        def generate(self, *args, **kwargs):
            return "ok"

        def generate_stream(self, *args, **kwargs):
            yield "ok"

    # Replace the actual TransformersModel with a dummy so we don't download models in tests
    monkeypatch.setattr(core, "TransformersModel", DummyModel)

    agent = core.Agent()

    assert isinstance(agent.model, DummyModel)
    assert agent.model.kwargs["model_id"] == os.environ.get("SMALL_MODEL_ID", "distilgpt2")
    assert agent.model.kwargs["max_new_tokens"] <= 512


def test_tokenizer_chat_template_fallback(monkeypatch):
    """If the tokenizer has no chat_template, Agent should set a fallback so
    Transformers chat helper functions don't raise the 'chat_template not set' error.
    """
    monkeypatch.setattr(core, "_is_low_memory", lambda threshold_gb=12: False)

    class DummyModel2:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            # tokenizer intentionally has no chat_template
            class Tok:
                pass

            self.tokenizer = Tok()

        class _StreamDelta:
            def __init__(self, content):
                self.content = content
                self.token_usage = None
                self.tool_calls = None

        def generate(self, *args, **kwargs):
            class _ChatMessage:
                def __init__(self, content):
                    self.content = content
                    self.token_usage = None

                def render_as_markdown(self):
                    return self.content

            return _ChatMessage("ok")

        def generate_stream(self, *args, **kwargs):
            # Yield objects mimicking the stream delta objects that
            # the agent expects (having at least a 'content' and
            # 'token_usage' attribute).
            yield self._StreamDelta("ok")

    monkeypatch.setattr(core, "TransformersModel", DummyModel2)

    agent = core.Agent()

    # tokenizer should now have a chat_template fallback
    tok = agent.model.tokenizer
    assert hasattr(tok, "chat_template")
    # ensure callable
    assert callable(tok.chat_template)

    # agent.chat should not raise
    agent.chat("hello")


def test_qwen_model_gets_qwen_fallback(monkeypatch):
    monkeypatch.setattr(core, "_is_low_memory", lambda threshold_gb=12: False)

    class DummyModelQwen:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            class Tok:
                pass
            self.tokenizer = Tok()

        def generate(self, *args, **kwargs):
            return "ok"

        def generate_stream(self, *args, **kwargs):
            yield type("Delta", (), {"content": "ok", "token_usage": None, "tool_calls": None})()

    monkeypatch.setenv("MODEL_ID", "Qwen/Qwen3-Next-80B-A3B-Thinking")
    monkeypatch.setattr(core, "TransformersModel", DummyModelQwen)

    agent = core.Agent()
    tok = agent.model.tokenizer
    assert hasattr(tok, "chat_template")
    # The Qwen fallback should produce role-prefixed lines
    out = tok.chat_template([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
    assert "User: Hi" in out or "user: Hi" in out


def test_template_compilation_error_triggers_fallback(monkeypatch, tmp_path):
    """If the model/runtime raises a template compilation error that asks to "use Qwen",
    the agent should retry with SMALL_CAUSAL_MODEL_ID."""
    # Force not low-memory so Agent will try the configured model first
    monkeypatch.setattr(core, "_is_low_memory", lambda threshold_gb=12: False)

    # Keep track of which model_id was instantiated
    instantiated = []

    class DummyModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instantiated.append(self.kwargs.get("model_id"))

        def generate(self, *args, **kwargs):
            return "ok"

        def generate_stream(self, *args, **kwargs):
            yield "ok"

    # A CodeAgent stub that raises on run if the model is the 'bad' one
    class DummyCodeAgent:
        def __init__(self, tools, model, stream_outputs=False, **kwargs):
            self.model = model
            self.stream_outputs = stream_outputs

        def run(self, task):
            if self.model.kwargs.get("model_id") == "badmodel":
                    # Simulate the runtime error raised by a model / template compiler
                    raise Exception("Can't compile non template nodes (please use Qwen)")
            # otherwise, succeed silently
            self.ran = True

    monkeypatch.setenv("MODEL_ID", "badmodel")
    monkeypatch.setenv("SMALL_CAUSAL_MODEL_ID", "goodsmall")
    monkeypatch.setattr(core, "TransformersModel", DummyModel)
    monkeypatch.setattr(core, "CodeAgent", DummyCodeAgent)

    agent = core.Agent()

    # Before calling chat, ensure first created model was 'badmodel'
    assert "badmodel" in instantiated

    # Should not raise; should retry with 'goodsmall'
    out = agent.chat("hello")
    assert "goodsmall" in instantiated
    assert "fallback" in out.lower()

