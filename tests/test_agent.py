"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, confirm_deferred_tools

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz"}
ALL_TOOLS = BASE_TOOLS | {"save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert ALL_TOOLS <= offered

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_learning_log_is_saved_only_after_testmodel_approval(monkeypatch, tmp_path):
    from pydantic_ai import DeferredToolRequests

    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr("app.agent.build_model", lambda: TestModel(call_tools=[]))
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending_result = agent.run_sync("学習記録を保存して", model=model)

    requests = pending_result.output
    assert isinstance(requests, DeferredToolRequests)
    assert len(requests.approvals) == 1
    assert not tools.LEARNING_LOG_DIR.exists()

    call = requests.approvals[0]
    result = confirm_deferred_tools(agent, pending_result)

    saved_files = list(tools.LEARNING_LOG_DIR.glob("*.md"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8") == call.args["content"]
    assert str(saved_files[0].relative_to(tmp_path)) in result.output


def test_learning_log_is_not_saved_when_testmodel_approval_is_denied(monkeypatch, tmp_path):
    from pydantic_ai import DeferredToolRequests

    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr("app.agent.build_model", lambda: TestModel(call_tools=[]))
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending_result = agent.run_sync("学習記録を保存して", model=model)

    requests = pending_result.output
    assert isinstance(requests, DeferredToolRequests)
    result = confirm_deferred_tools(agent, pending_result)

    assert isinstance(result.output, str)
    assert not tools.LEARNING_LOG_DIR.exists()
