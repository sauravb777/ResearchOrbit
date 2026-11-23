from __future__ import annotations
from typing import Any, TypedDict
class ResearchChatState(TypedDict, total=False):
    question: str
    intent: str
    items: list[dict]
    answer: str
    used_ids: list[str]
    llm_used: bool
    verification_used: bool
    failure: str
    references: list[dict]
    source_ids: list[str]
    using_langgraph: bool
    using_langchain: bool
def _run_fallback(nodes: list[tuple[str, Any]], state: dict) -> dict:
    for _, fn in nodes:
        state = {**state, **(fn(state) or {})}
    return state
def _build(nodes: dict[str, Any]):
    try:
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(ResearchChatState)
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        names = list(nodes)
        graph.add_edge(START, names[0])
        for a, b in zip(names, names[1:]):
            graph.add_edge(a, b)
        graph.add_edge(names[-1], END)
        return graph.compile(), True
    except Exception:
        return None, False
def _langchain_invoke(fn, payload):
    """Run one callable through LangChain when available, otherwise directly."""
    try:
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(fn).invoke(payload), True
    except Exception:
        return fn(payload), False
def run_research_chat_graph(*, question: str, route_fn, retrieve_fn, draft_fn, verify_fn, provenance_fn, cancel_check=None) -> dict:
    """Execute the complete chat chain with one bounded state object.
    Each injected function returns a partial state dictionary.  Verification is
    always part of the path when a draft exists; it is not exposed as a second
    user action.  Retrieval and provenance are deterministic application logic.
    """
    def _check():
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Chat cancelled")
    def route(s):
        _check()
        out, lc = _langchain_invoke(lambda x: {"intent": route_fn(x["question"])}, s)
        return {**(out or {}), "using_langchain": bool(lc)}
    def retrieve(s):
        _check()
        out = retrieve_fn(s)
        _check()
        return out
    def draft(s):
        _check()
        out, lc = _langchain_invoke(draft_fn, s)
        _check()
        return {**(out or {}), "using_langchain": bool(s.get("using_langchain") or lc)}
    def verify(s):
        _check()
        out = verify_fn(s)
        _check()
        return out
    def provenance(s):
        _check()
        return provenance_fn(s)
    nodes = {"route": route, "retrieve": retrieve, "draft": draft, "verify": verify, "provenance": provenance}
    graph, using_langgraph = _build(nodes)
    initial = {"question": question}
    if graph is not None:
        try:
            out = graph.invoke(initial)
            if isinstance(out, dict) and "answer" in out:
                out["using_langgraph"] = True
                return out
        except Exception:
            pass
    out = _run_fallback(list(nodes.items()), initial)
    out["using_langgraph"] = False
    return out
