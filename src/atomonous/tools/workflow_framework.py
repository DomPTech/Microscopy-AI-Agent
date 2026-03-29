import abc
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class WorkflowState(BaseModel):
    """
    State passed between nodes in the workflow.
    Ensures type safety and data integrity across experimental steps.
    """
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Input and tuning parameters.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Metadata and context from AI or previous steps.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Output data, such as images, arrays, or JSON.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Quantitative measurements for success evaluation.")
    history: List[str] = Field(default_factory=list, description="Log of executed steps and significant events.")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during execution.")

class WorkflowNode(abc.ABC):
    """
    Abstract base class for all nodes in the workflow graph.
    """
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abc.abstractmethod
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        """
        Execute the node's logic. Must return the updated state.
        """
        pass

class WorkflowTemplate(BaseModel):
    """
    Represents a parsed YAML workflow graph.
    """
    name: str = Field(..., description="Name of the workflow.")
    description: str = Field("", description="Description of the workflow.")
    nodes: List[Dict[str, Any]] = Field(..., description="List of node configurations. Each must have at least 'id' and 'type'.")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="List of edges, e.g. {'source': 'nodeA', 'target': 'nodeB', 'label': 'then', 'style': 'dotted'}.")

class WorkflowExecutor:
    """
    Executes a WorkflowTemplate by orchestrating WorkflowNodes.
    """
    def __init__(self, template: WorkflowTemplate, node_registry: Dict[str, type]):
        self.template = template
        self.node_registry = node_registry

    def _topological_sort(self) -> List[Dict[str, Any]]:
        """
        Sort nodes topologically based on edges.
        """
        in_degree = {n["id"]: 0 for n in self.template.nodes}
        adj_list = {n["id"]: [] for n in self.template.nodes}
        node_map = {n["id"]: n for n in self.template.nodes}

        for edge in self.template.edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src and tgt and src in in_degree and tgt in in_degree:
                adj_list[src].append(tgt)
                in_degree[tgt] += 1

        queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            curr = queue.pop(0)
            sorted_nodes.append(node_map[curr])
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(self.template.nodes):
            raise ValueError("Cycle detected in workflow graph or invalid edges.")

        return sorted_nodes

    def run(self, initial_state: Optional[WorkflowState] = None, context: Optional[dict] = None) -> WorkflowState:
        state = initial_state or WorkflowState()
        state.history.append(f"Starting workflow: {self.template.name}")
        
        context = context or {}

        try:
            sorted_node_configs = self._topological_sort()
        except Exception as e:
            state.errors.append(f"Failed to compile workflow: {str(e)}")
            return state

        for config in sorted_node_configs:
            node_id = config.get("id")
            node_type = config.get("type")
            node_params = config.get("params", {})

            if node_type not in self.node_registry:
                state.errors.append(f"Node execution failed: Unknown node type '{node_type}' for node '{node_id}'.")
                break

            try:
                # Instantiate and run
                node_class = self.node_registry[node_type]
                node_instance = node_class(name=node_id, **node_params)
                
                print(f"\\n--- Executing Node: {node_id} ({node_type}) ---")
                state.history.append(f"Executing node '{node_id}' of type '{node_type}'...")
                
                # Execute the node
                state = node_instance.execute(state, context=context)
                
                # Check if node deliberately raised error in state
                if state.errors and state.errors[-1].startswith("FATAL"):
                    break
            except Exception as e:
                state.errors.append(f"Exception in node '{node_id}' ({node_type}): {str(e)}")
                break

        state.history.append("Workflow execution completed.")
        return state
