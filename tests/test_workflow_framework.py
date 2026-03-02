import pytest
import os
import yaml
from app.tools.workflow_framework import WorkflowState, WorkflowNode, WorkflowTemplate, WorkflowExecutor
from app.tools.microscopy import NODE_REGISTRY, design_workflow

def test_workflow_execution():
    # Construct a real yaml test
    yaml_content = """
name: "TestWorkflow"
description: "A standard test workflow"
nodes:
  - id: "ctx"
    type: "AIContext"
    params:
      query: "test"
  - id: "tool"
    type: "MicroscopeTool"
    params:
      tool: "get_microscope_state"
edges:
  - source: "ctx"
    target: "tool"
"""
    parsed_yaml = yaml.safe_load(yaml_content)
    template = WorkflowTemplate(**parsed_yaml)
    executor = WorkflowExecutor(template, NODE_REGISTRY)

    final_state = executor.run()
    
    # Validation
    assert "ctx" in final_state.context
    assert final_state.errors == []
    assert len(final_state.history) > 0
    assert "tool" in final_state.data

def test_design_workflow():
    yaml_content = """
name: "ComplexWorkflow"
description: "A complex workflow matching the visual structure"
nodes:
  - id: "__start__"
    type: "AIContext"
    params: { query: "Start workflow" }
  - id: "initialize"
    type: "AIContext"
    params: { query: "Init" }
  - id: "validate"
    type: "AIQuality"
    params: { evaluate_node: "initialize" }
  - id: "recommend"
    type: "AIContext"
    params: { query: "Recommend params" }
  - id: "confirm"
    type: "AIQuality"
    params: { evaluate_node: "recommend" }
  - id: "acquire"
    type: "MicroscopeTool"
    params: { tool: "capture_image", args: {} }
  - id: "assess"
    type: "AIQuality"
    params: { evaluate_node: "acquire" }
  - id: "align"
    type: "MicroscopeTool"
    params: { tool: "place_beam", args: {"x": 0.5, "y": 0.5} }
  - id: "optimize"
    type: "MicroscopeTool"
    params: { tool: "tune_C1A1", args: {} }
  - id: "analyze"
    type: "AIContext"
    params: { query: "Analyze results" }
  - id: "__end__"
    type: "AIContext"
    params: { query: "End workflow" }
edges:
  - source: "__start__"
    target: "initialize"
  - source: "initialize"
    target: "validate"
  - source: "validate"
    target: "recommend"
    label: "then"
    style: "dotted"
  - source: "validate"
    target: "__end__"
    label: "otherwise"
    style: "dotted"
  - source: "recommend"
    target: "confirm"
  - source: "confirm"
    target: "acquire"
  - source: "confirm"
    target: "optimize"
    label: "then"
    style: "dotted"
  - source: "acquire"
    target: "assess"
    label: "then"
    style: "dotted"
  - source: "assess"
    target: "align"
    label: "then"
    style: "dotted"
  - source: "align"
    target: "acquire"
  - source: "assess"
    target: "optimize"
    label: "otherwise"
    style: "dotted"
  - source: "optimize"
    target: "analyze"
    label: "otherwise"
    style: "dotted"
  - source: "analyze"
    target: "__end__"
"""
    result = design_workflow(name="test_png_output", yaml_content=yaml_content)
    assert "Successfully designed workflow" in result
    
    # Check if files are created (yaml and png)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflows_dir = os.path.join(base_dir, "app", "workflows")
    
    assert os.path.exists(os.path.join(workflows_dir, "test_png_output.yaml"))
    assert os.path.exists(os.path.join(workflows_dir, "test_png_output.png"))
