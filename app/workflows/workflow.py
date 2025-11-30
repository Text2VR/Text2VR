"""Workflow assembly for the LangGraph panorama pipeline."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .steps.generation import (
    panorama_generation_node,
    query_rewrite_node,
)
from .steps.segmentation import segmentation_node
from .steps.three_d import (
    asset_3d_generation_node,
    ply_generation_node,
)
from .steps.inpainting import inpainting_node
from ..models.workflow_state import WorkflowState


def create_workflow():
    """Compile and return the LangGraph workflow for panorama generation."""

    workflow = StateGraph(WorkflowState)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("panorama_generation", panorama_generation_node)
    workflow.add_node("segmentation", segmentation_node)
    workflow.add_node("asset_3d_generation", asset_3d_generation_node)
    workflow.add_node("inpainting", inpainting_node)
    workflow.add_node("ply_generation", ply_generation_node)

    workflow.set_entry_point("query_rewrite")
    workflow.add_edge("query_rewrite", "panorama_generation")
    workflow.add_edge("panorama_generation", "segmentation")
    workflow.add_edge("segmentation", "asset_3d_generation")
    workflow.add_edge("asset_3d_generation", "inpainting")
    workflow.add_edge("inpainting", "ply_generation")
    workflow.add_edge("ply_generation", END)

    return workflow.compile()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workflow.py <user_input>")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    workflow = create_workflow()

    initial_state = {
        "user_input": user_input,
        "rewritten_query": "",
        "scene_name": "",
        "panorama_path": "",
        "segmentation_data": {},
        "inpainted_panorama_path": "",
        "ply_path": "",
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
    print(f"🎨 Inpainted: {result.get('inpainted_panorama_path', 'N/A')}")
    print(f"🎲 PLY: {result.get('ply_path', 'N/A')}")
