"""State definitions for the LangGraph panorama workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class WorkflowState(TypedDict):
    """Represents the shared state that flows through the LangGraph workflow."""

    task_id: str
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str
    messages: Annotated[List[BaseMessage], operator.add]
