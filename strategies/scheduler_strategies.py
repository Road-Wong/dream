# scheduler_strategies.py
from typing import Dict, List, Optional, Callable
from collections import defaultdict, deque
import simpy # simpy might be needed for env.now in some strategies

class SchedulerStrategy:
    """Base class for scheduling strategies."""
    
    def __init__(self, env: simpy.Environment, nodes: List[Dict], tasks: List[Dict], 
                 dependencies: List[Dict], node_resources: Dict, node_schedulers: Dict,
                 log_manager=None, get_node_current_status_fn=None, get_task_state_fn=None):
        self.env = env
        self.nodes_spec = nodes
        self.tasks_spec = tasks
        self.dependencies_spec = dependencies
        self.node_resources = node_resources       # Dict[str, NodeResourceManager]
        self.node_schedulers = node_schedulers     # Dict[str, NodeScheduler]
        self.log_manager = log_manager
        self.get_node_current_status = get_node_current_status_fn # Callback for dynamic node status
        self.get_task_state = get_task_state_fn # Callback for task states

        self.task_map = {task['id']: task for task in self.tasks_spec}
        
        self.task_deps = defaultdict(list)
        self.task_reverse_deps = defaultdict(list)
        for dep in self.dependencies_spec:
            self.task_deps[dep['source']].append(dep['target'])
            self.task_reverse_deps[dep['target']].append(dep['source'])
            
    def topological_sort(self) -> List[str]:
        in_degree = {task['id']: 0 for task in self.tasks_spec}
        for dep in self.dependencies_spec:
            in_degree[dep['target']] += 1
        
        queue = deque([task['id'] for task in self.tasks_spec if in_degree[task['id']] == 0])
        topo_order = []
        
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            for successor in self.task_deps.get(current, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        if len(topo_order) != len(self.tasks_spec):
            msg = "Warning: Task graph may contain a cycle or be disconnected!"
            if self.log_manager: self.log_manager.log_print(msg)
            else: print(msg)
        return topo_order
    
    def find_earliest_completion_times(self) -> Dict[str, float]:
        """Estimates earliest completion time based on task spec (duration hint)."""
        earliest_completion = {}
        for task_id in self.topological_sort():
            task_spec = self.task_map[task_id]
            # Use duration hint from task_spec if available
            task_duration_hint = task_spec.get('duration', 1.0) 
            
            if not self.task_reverse_deps.get(task_id, []): # Source task
                earliest_completion[task_id] = task_duration_hint
            else:
                max_pred_completion = 0
                for pred_id in self.task_reverse_deps[task_id]:
                    max_pred_completion = max(max_pred_completion, earliest_completion.get(pred_id, 0))
                earliest_completion[task_id] = max_pred_completion + task_duration_hint
        return earliest_completion

    def find_suitable_node(self, task_spec: Dict) -> Optional[Dict]:
        """
        Finds a suitable node for the task. Must be implemented by subclasses.
        Should return a node_spec dictionary from self.nodes_spec, or None.
        """
        raise NotImplementedError("Subclasses must implement find_suitable_node.")

    # start_task logic is usually handled by ClusterManager after node selection,
    # but a strategy could override parts of it if needed.
    # For now, strategies focus on find_suitable_node.