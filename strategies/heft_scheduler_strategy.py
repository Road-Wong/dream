# heft_scheduler_strategy.py
from typing import Dict, List, Optional, Tuple
import simpy
from collections import defaultdict

from strategies.scheduler_strategies import SchedulerStrategy
# from node_resource_manager import NodeResourceManager # For type hinting if needed

class HEFTScheduler(SchedulerStrategy):
    """
    HEFT (Heterogeneous Earliest-Finish-Time) inspired scheduling strategy.
    It prioritizes tasks based on their upward rank (criticality) and
    selects the node that offers the earliest finish time (EFT) for the task.
    """
    def __init__(self, env: simpy.Environment, nodes: List[Dict], tasks: List[Dict],
                 dependencies: List[Dict], node_resources: Dict, node_schedulers: Dict,
                 log_manager=None, get_node_current_status_fn=None, get_task_state_fn=None):
        super().__init__(env, nodes, tasks, dependencies, node_resources, node_schedulers,
                         log_manager, get_node_current_status_fn, get_task_state_fn)
        
        self.task_avg_exec_time: Dict[str, float] = {} # Store average execution time for rank calculation
        self.task_upward_ranks: Dict[str, float] = {}
        self._calculate_avg_exec_times()
        self._calculate_upward_ranks() # Similar to ZhaoCPC's rank calculation

        if self.log_manager and self.log_manager.verbose:
            sorted_ranks = sorted(self.task_upward_ranks.items(), key=lambda item: item[1], reverse=True)
            self.log_manager.log_print("HEFT: Calculated Upward Ranks (higher is more critical):")
            for task_id, rank in sorted_ranks:
                self.log_manager.log_print(f"  Task {task_id}: {rank:.2f}")

    def _calculate_avg_exec_times(self):
        """
        Calculates average execution time for each task.
        For HEFT, this often involves averaging execution time across all processors,
        or using a pre-defined computation cost. We'll use the 'duration' from task_spec.
        """
        for task_spec in self.tasks_spec:
            # A more standard HEFT might calculate avg_w_i = sum(p_ij for j in processors) / num_processors
            # For simplicity, we use the provided 'duration' as the average computation cost.
            self.task_avg_exec_time[task_spec['id']] = task_spec.get('duration', 1.0)

    def _compute_rank_u(self, task_id: str, 
                        avg_processing_times: Dict[str, float],
                        successors: Dict[str, List[str]], 
                        memo: Dict[str, float]) -> float:
        """
        Recursively computes the upward rank of a task.
        rank_u(t_i) = avg_w_i + max_{t_j \\in succ(t_i)} (avg_comm_ij + rank_u(t_j))
        Communication costs (avg_comm_ij) are simplified / ignored in this implementation
        to align with how ClusterManager handles data transfer separately after node assignment.
        So, rank_u(t_i) = avg_w_i + max_{t_j \\in succ(t_i)} (rank_u(t_j))
        """
        if task_id in memo:
            return memo[task_id]

        avg_w_i = avg_processing_times.get(task_id, 1.0) 

        max_succ_component = 0.0
        if task_id in successors and successors[task_id]:
            for succ_task_id in successors[task_id]:
                # avg_comm_cost = 0 # Simplified: Ignoring average communication cost in rank
                                  # Actual communication is handled by NetworkSimulator later.
                                  # A full HEFT would estimate this.
                max_succ_component = max(max_succ_component, 
                                         self._compute_rank_u(succ_task_id, avg_processing_times, successors, memo))
        
        memo[task_id] = avg_w_i + max_succ_component
        return memo[task_id]

    def _calculate_upward_ranks(self):
        """Calculates upward ranks for all tasks."""
        for task_spec in self.tasks_spec:
            task_id = task_spec['id']
            if task_id not in self.task_upward_ranks:
                self._compute_rank_u(task_id, self.task_avg_exec_time, self.task_deps, self.task_upward_ranks)

    def _estimate_earliest_finish_time(self, task_spec: Dict, node_data: Dict) -> float:
        """
        Estimates EFT of a task on a node.
        EFT(task_i, node_j) = max(ready_time(task_i, node_j), avail_time(node_j)) + exec_time(task_i, node_j)
        - ready_time: Time when all input data for task_i has arrived at node_j.
                      This is complex as it depends on parent task completions and locations.
                      Simplified: Assume data is ready by env.now if dependencies are met
                                  (ClusterManager handles actual data transfer waits).
        - avail_time(node_j): Time when node_j becomes free.
                              Simplified: env.now if NodeResourceManager.has_available_resources is true.
                                          Otherwise, this requires peeking into NodeScheduler's queue,
                                          which is not directly exposed here.
        - exec_time: Execution time of task_i on node_j. We use task_spec['duration'].
        """
        # Simplified: task_spec['duration'] is used as exec_time on any node.
        # A more accurate HEFT would use p_ij (exec time of task i on processor j).
        exec_time_on_node = task_spec.get('duration', 1.0) 

        # Simplified availability and ready time:
        # If node has resources now, assume it can start now.
        # This is a common simplification when NodeScheduler details are not deeply integrated into strategy.
        # Actual queueing and data transfer delays are handled by ClusterManager and NodeScheduler.
        
        # avail_time approximation:
        # For this simplified HEFT, we assume that if `has_available_resources` (checked by caller)
        # is true, the node is available "now" or very soon.
        # The actual start time will be determined by the NodeScheduler's queue.
        # Our EFT here is more of a "potential earliest finish if started now without queueing on this node".
        
        # ready_time approximation:
        # Assume if task is being considered, its parent data will be made available.
        # ClusterManager's _process_task_completion and _wait_for_transfers handles this.
        
        # So, EFT = current_time + execution_time_on_this_node
        # This is a common way to implement MCT or HEFT's processor selection phase in simulators
        # where the full state of all node queues isn't easily queryable by the strategy.
        return self.env.now + exec_time_on_node


    def find_suitable_node(self, task_spec: Dict) -> Optional[Dict]:
        """
        Finds a suitable node for the task.
        It considers all nodes that can run the task and picks the one
        that offers the earliest estimated finish time (EFT).
        Ties are broken by choosing node with higher compute power or lower ID.
        
        Note: Task prioritization (picking which task to schedule next from ready_queue)
        is typically handled by ClusterManager using the upward_ranks from this strategy.
        This method is purely for node selection for a *given* task.
        """
        task_id = task_spec['id']
        
        candidate_nodes_eft: List[Tuple[float, float, str, Dict]] = [] # (eft, node_compute_power, node_id, node_data)

        required_flops = task_spec['compute_demand']
        required_memory_mb = task_spec['memory_demand'] * 1024
        required_model = task_spec['model_required']
        
        if self.log_manager and self.log_manager.verbose:
            self.log_manager.log_print(f"HEFT: Finding node for task {task_id} (UpwardRank: {self.task_upward_ranks.get(task_id, 0.0):.2f})")

        for node_data in self.nodes_spec:
            node_id = node_data['id']
            res_man = self.node_resources[node_id]

            # Check 1: Basic capability (model, total capacity)
            exceeds_limits, reason = res_man.exceeds_system_limits(
                compute_flops=required_flops,
                memory_mb=required_memory_mb,
                model_name=required_model
            )
            if exceeds_limits:
                if self.log_manager and self.log_manager.verbose:
                     self.log_manager.log_print(f"HEFT: Node {node_id} rejected {task_id} (limits check). Reason: {reason}")
                continue

            # Check 2: Current resource availability (for our simplified EFT)
            # A full HEFT would estimate EFT even if resources are not currently free by looking at queue.
            # Our simplification: if not available now, it's not a candidate for *immediate* scheduling by this logic.
            if not res_man.has_available_resources(
                compute_flops=required_flops,
                memory_mb=required_memory_mb
            ):
                if self.log_manager and self.log_manager.verbose:
                     self.log_manager.log_print(f"HEFT: Node {node_id} rejected {task_id} (no current resources for simplified EFT).")
                continue
            
            est_finish_time = self._estimate_earliest_finish_time(task_spec, node_data)
            
            candidate_nodes_eft.append((est_finish_time, node_data['compute_power'], node_id, node_data))

        if not candidate_nodes_eft:
            if self.log_manager:
                self.log_manager.log_print(f"HEFT: No suitable node found for task {task_id} based on current availability and simplified EFT.")
            return None

        # Sort candidates:
        # 1. Primary: Earliest Finish Time (ascending)
        # 2. Secondary: Node Compute Power (descending - for tie-breaking, prefer more powerful node)
        # 3. Tertiary: Node ID (ascending - for deterministic tie-breaking)
        candidate_nodes_eft.sort(key=lambda x: (x[0], -x[1], x[2]))
        
        best_node_spec = candidate_nodes_eft[0][3] 
        
        if self.log_manager:
            self.log_manager.log_print(f"HEFT: Task {task_id} assigned to node {best_node_spec['id']} (EFT: {candidate_nodes_eft[0][0]:.2f}, Rank: {self.task_upward_ranks.get(task_id,0.0):.2f})")
        
        return best_node_spec