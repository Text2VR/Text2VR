"""Compatibility layer for the refactored LangGraph workflow modules."""

from __future__ import annotations

from .nodes import (
    panorama_generation_node,
    query_rewrite_node,
    segmentation_node,
)
from .states import WorkflowState
from .workflow import create_workflow

__all__ = [
    "WorkflowState",
    "create_workflow",
    "query_rewrite_node",
    "panorama_generation_node",
    "segmentation_node",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python langgraph_workflow.py <user_input>")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    workflow = create_workflow()

    initial_state = {
        "user_input": user_input,
        "rewritten_query": "",
        "scene_name": "",
        "panorama_path": "",
        "segmentation_data": {},
        "messages": [],
    }

    print(f"🚀 Starting workflow for: '{user_input}'")
    result = workflow.invoke(initial_state)

    print(f"\n📝 Original: {result['user_input']}")
    print(f"✨ Rewritten: {result['rewritten_query']}")
    print(f"🎬 Scene: {result['scene_name']}")
    print(f"🖼️ Panorama: {result['panorama_path']}")
    segmentation_summary = (
        list(result["segmentation_data"].get("prompts", {}).keys())
        if result.get("segmentation_data")
        else "No objects found"
    )
    print(f"🔍 Segmentation: {segmentation_summary}")
