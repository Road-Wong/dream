# cluster_manager.py
import simpy
import time as _time
import functools
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque
import heapq
import copy
import importlib

from core.node_scheduler import NodeScheduler
from core.node_resource_manager import NodeResourceManager
from core.network_sim import NetworkSimulator
from core.log_manager import LogManager
from generators.task_generator import get_duration_by_model

# Import strategies dynamically based on strategy name
STRATEGY_MODULE_MAP = {
    'dream_proactive': 'dream_proactive_scheduler',
    'heft': 'heft_scheduler_strategy',
}
STRATEGY_CLASS_MAP = {
    'dream_proactive': 'DREAMProactiveScheduler',
    'heft': 'HEFTScheduler',
}


class ClusterManager:
    def __init__(self, env: simpy.Environment, network: Dict, task_dag: Dict, 
                 log_manager: LogManager, network_simulator: NetworkSimulator, 
                 scheduler_strategy: str = 'greedy', 
                 # P1-3 FIX: Adjusted default weights — w_imm 0.4→0.38, w_opp 0.15→0.18
                 # to strengthen C_opp (sole reservation channel after P0-1 fix)
                 w_immediate: float = 0.38, 
                 w_ripple: float = 0.3, 
                 w_stability: float = 0.15, 
                 w_opportunity: float = 0.18,
                 scheduler_config: Optional[Dict] = None,
                 noise_injector=None):
        self.env = env
        self.log_manager = log_manager
        self.network_simulator = network_simulator
        self.simulation_start_time = -1.0 # Initialize to indicate not yet started
        
        # Store DREAM weights for use in scheduler initialization
        self.w_immediate = w_immediate
        self.w_ripple = w_ripple
        self.w_stability = w_stability
        self.w_opportunity = w_opportunity
        self.scheduler_config = scheduler_config or {}
        self.noise_injector = noise_injector

        self.nodes_spec = network['nodes'] # Renamed from self.nodes to avoid conflict
        self.tasks_spec = task_dag['tasks'] # Renamed from self.tasks
        self.dependencies_spec = task_dag['dependencies'] # Renamed

        self.scheduler_tasks_spec: List[Dict] = []
        for task_data in self.tasks_spec:
            scheduler_task_data = dict(task_data)
            if self.noise_injector is not None:
                try:
                    scheduler_task_data = self.noise_injector.apply_to_task_spec(scheduler_task_data)
                except Exception:
                    scheduler_task_data = dict(task_data)
            self.scheduler_tasks_spec.append(scheduler_task_data)
        self.scheduler_task_map = {task['id']: task for task in self.scheduler_tasks_spec}

        self.node_resources: Dict[str, NodeResourceManager] = {}
        self.node_schedulers: Dict[str, NodeScheduler] = {}
        for node_data in self.nodes_spec:
            node_id = node_data['id']
            res_man = NodeResourceManager(
                total_flops=node_data['compute_power'],
                total_memory=node_data['memory'] * 1024, # GB to MB
                available_models=node_data['models'],
                node_id=node_id
            )
            self.node_resources[node_id] = res_man
            self.node_schedulers[node_id] = NodeScheduler(self.env, res_man, self.log_manager, node_id)

        # Pre-compute max compute power for task timeout estimation
        self._max_compute_power = max((n.get('compute_power', 0) for n in self.nodes_spec), default=1.0)
        if self._max_compute_power <= 0:
            self._max_compute_power = 1.0
        self.task_timeout_multiplier: float = scheduler_config.get('task_timeout_multiplier', 5.0) if scheduler_config else 5.0

        self.task_states: Dict[str, Dict] = {}
        for task_data in self.tasks_spec:
            self.task_states[task_data['id']] = {
                'id': task_data['id'],
                'required_flops': task_data['compute_demand'],
                'required_memory': task_data['memory_demand'], # GB
                'required_model': task_data['model_required'],
                'data_size': task_data.get('data_size', 0),
                'duration_hint': task_data.get('duration', get_duration_by_model(task_data['model_required'])),
                'original_id': task_data.get('original_id', task_data['id']), # For dynamic DAGs
                'creation_time': 0.0, # Will be updated
                'start_time': None,
                'completion_time': None,
                'failure_reason': None,
                'node_assigned': None,
                'status': 'pending', # pending, ready, running, completed, failed
                'priority': task_data.get('priority', 'normal')
            }
            # Compute per-task timeout budget based on task characteristics.
            # Scheduling timeout = max(estimated_exec_time, min_floor) × multiplier
            # This bounds how long a task may wait in the ready queue before being
            # declared failed (triggers cascade failure for DAG successors).
            estimated_exec = task_data.get('duration', get_duration_by_model(task_data['model_required']))
            min_timeout_floor = 3.0  # Minimum 3s for scheduling delay + retries
            task_timeout = max(estimated_exec, min_timeout_floor) * self.task_timeout_multiplier
            # Cap at a maximum to prevent very long-lived tasks from blocking
            task_timeout = min(task_timeout, 120.0)
            self.task_states[task_data['id']]['task_timeout'] = task_timeout
        
        self.task_map = {task['id']: task for task in self.tasks_spec} # For quick lookup of spec

        self.task_deps = defaultdict(list)
        self.task_reverse_deps = defaultdict(list)
        for dep in self.dependencies_spec:
            self.task_deps[dep['source']].append(dep['target'])
            self.task_reverse_deps[dep['target']].append(dep['source'])

        # OPT: pending_dep_count for O(1) dependency check instead of iterating reverse_deps
        self.pending_dep_count: Dict[str, int] = {}
        for task_data in self.tasks_spec:
            tid = task_data['id']
            self.pending_dep_count[tid] = len(self.task_reverse_deps.get(tid, []))

        self.task_assignments: Dict[str, str] = {} # task_id -> node_id
        self.node_completion_times = {node['id']: 0.0 for node in self.nodes_spec}
        self.node_running_tasks: Dict[str, set] = {node['id']: set() for node in self.nodes_spec}  # OPT: set for O(1) add/remove
        
        self.currently_running_tasks = set()
        self.completed_tasks = set()
        self.failed_tasks_details: List[Dict] = [] # Store dicts with failure info

        # Task timeout mechanism: each task gets a timeout budget based on its
        # estimated execution time. If a task remains unscheduled beyond its
        # timeout, it is marked as failed, and its DAG successors become
        # permanently blocked (cascade failure).
        self.task_ready_time: Dict[str, float] = {}  # task_id -> first ready timestamp

        self.data_transfer_events: Dict[str, simpy.Event] = {} # "src->tgt" -> event
        self.first_task_node: Optional[str] = None # For final data return if needed (not fully used yet)
        self.initial_task_nodes: Dict[str, str] = {} # task_id -> pre-assigned node_id

        self.decision_time_records = []
        self._init_scheduler_strategy(scheduler_strategy)
        
        self.ready_to_schedule_tasks = set()  # OPT: set for O(1) in/add/remove

    def _init_scheduler_strategy(self, strategy_name: str):
        """Unified strategy initialization with inspect-based parameter passing.
        
        All strategies share the same base parameters. Strategy-specific parameters
        (config, network_simulator) are passed only when the strategy's __init__
        accepts them, ensuring fair experimental environment between DREAM and HEFT.
        """
        import inspect
        
        if strategy_name.lower() not in STRATEGY_MODULE_MAP:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available strategies: {list(STRATEGY_MODULE_MAP.keys())}"
            )
        
        module_name = STRATEGY_MODULE_MAP[strategy_name.lower()]
        class_name = STRATEGY_CLASS_MAP[strategy_name.lower()]
        
        strategy_module = importlib.import_module(f"strategies.{module_name}")
        strategy_class = getattr(strategy_module, class_name)

        # Build common kwargs shared by all strategies
        common_kwargs = {
            'env': self.env,
            'nodes': self.nodes_spec,
            'tasks': self.scheduler_tasks_spec,
            'dependencies': self.dependencies_spec,
            'node_resources': self.node_resources,
            'node_schedulers': self.node_schedulers,
            'log_manager': self.log_manager,
            'get_node_current_status_fn': lambda node_id: self.node_resources[node_id].get_resource_status(current_time=self.env.now),
            'get_task_state_fn': lambda task_id, default=None: self.task_states.get(task_id, default or {}),
        }
        
        # Add strategy-specific kwargs only if the strategy accepts them
        sig = inspect.signature(strategy_class.__init__)
        
        if 'config' in sig.parameters:
            # Build DREAM-specific config
            config = {
                'w_immediate': self.w_immediate,
                'w_ripple': self.w_ripple,
                'w_stability': self.w_stability,
                'w_opportunity': self.w_opportunity,
                'cpls_depth': 4,
                'cpls_branching': 3,
                'criticality_alpha': 1.8,
                'successor_mode': 'max_rank',
                'use_mean_std_threshold': True,
                'enable_ocap_normalization': True,
                'use_network_tt': True,
                'bandit_enabled': False,
                'cpls_soft_reservation': False,
                'eft_tolerance': 0.25,
                'reservation_guard_factor': 0.1,
                'lambda_penalty': 5.0,
                'criticality_threshold_ratio': 0.85,
                'priority_weights': {
                    'critical': 1.5,
                    'high': 1.2,
                    'medium': 1.0,
                    'normal': 0.9,
                    'low': 0.7
                }
            }
            if self.scheduler_config:
                config.update(self.scheduler_config)
            common_kwargs['config'] = config
        
        if 'network_simulator' in sig.parameters:
            common_kwargs['network_simulator'] = self.network_simulator

        self.scheduler = strategy_class(**common_kwargs)
        
        if self.log_manager.verbose:
            self.log_manager.log_print(f"Using scheduler strategy: {strategy_class.__name__}")

    def set_initial_task_node(self, task_id: str, node_id: str):
        if task_id in self.task_map and node_id in self.node_resources:
            self.initial_task_nodes[task_id] = node_id
            # Also pre-assign it in task_assignments, so find_suitable_node can respect it
            self.task_assignments[task_id] = node_id
            if self.log_manager.verbose:
                self.log_manager.log_print(f"Task {task_id} pre-assigned to node {node_id}.")
            return True
        if self.log_manager.verbose:
            self.log_manager.log_print(f"ERROR: Cannot pre-assign task {task_id} to node {node_id}.")
        return False

    def topological_sort(self) -> List[str]:
        return self.scheduler.topological_sort()
    
    def find_earliest_completion_times(self) -> Dict[str, float]:
        # This might be less relevant if not using a HEFT-like global schedule view
        return self.scheduler.find_earliest_completion_times()
        
    def _retry_task_later(self, task_id: str, delay: float = 1.0):
        """延迟一段时间后重新尝试调度任务
        
        Args:
            task_id: 要重新调度的任务ID
            delay: 延迟时间（秒）
        """
        yield self.env.timeout(delay)
        # 任务可能已经在其他地方被处理，先检查状态
        task_state = self.task_states.get(task_id, {})
        if task_state.get('status') in ['ready', 'pending']:
            # --- Timeout check: primary failure mechanism ---
            ready_time = self.task_ready_time.get(task_id, self.env.now)
            elapsed_since_ready = self.env.now - ready_time
            task_timeout = task_state.get('task_timeout', 50.0)
            if elapsed_since_ready > task_timeout:
                if self.log_manager.verbose:
                    self.log_manager.log_print(f"[{self.env.now:.2f}] 任务 {task_id} 超时 (elapsed={elapsed_since_ready:.1f}s > timeout={task_timeout:.1f}s)，标记为失败")
                self._fail_task(task_id, None, f"任务超时 (等待{elapsed_since_ready:.1f}s > 预算{task_timeout:.1f}s)")
                self.ready_to_schedule_tasks.discard(task_id)
                return

            # 尝试找到合适的节点重新调度
            node_id = self.find_suitable_node(task_id)
            if node_id:
                if self.log_manager.verbose:
                    self.log_manager.log_print(f"[{self.env.now:.2f}] 重新尝试调度任务 {task_id} 到节点 {node_id}")
                self.start_task(task_id, node_id)
            else:
                # 如果仍然找不到合适节点，延迟更长时间后再次尝试
                retry_count = task_state.get('retry_count', 0) + 1
                task_state['retry_count'] = retry_count
                # 读取调度器重试配置（若存在）—— 重试次数作为安全阀
                sched_retry_cfg = getattr(self.scheduler, 'retry_config', None)
                max_retries = sched_retry_cfg.get('max_retry_attempts', 100) if sched_retry_cfg else 100
                base_interval = sched_retry_cfg.get('retry_interval', 1.0) if sched_retry_cfg else 1.0
                if retry_count <= max_retries:
                    next_delay = min(max(base_interval, delay) * 1.5, 10.0)  # 渐进延迟，封顶10秒
                    if self.log_manager.verbose:
                        self.log_manager.log_print(f"[{self.env.now:.2f}] 任务 {task_id} 没有找到合适节点，{next_delay}秒后重试 (尝试 {retry_count}/{max_retries})")
                    self.env.process(self._retry_task_later(task_id, next_delay))
                else:
                    # 超过最大重试次数（安全阀），标记为失败
                    if self.log_manager.verbose:
                        self.log_manager.log_print(f"[{self.env.now:.2f}] 任务 {task_id} 超过最大重试次数 ({max_retries})，标记为失败")
                    self._fail_task(task_id, None, "超过最大重试次数，无法找到合适资源")
                    self.ready_to_schedule_tasks.discard(task_id)

    def find_suitable_node(self, task_id: str) -> Optional[str]: # Returns node_id
        task_spec = self.task_map[task_id]
        scheduler_task_spec = self.scheduler_task_map.get(task_id, task_spec)
         # If pre-assigned, use that node if still suitable (basic check)
        if task_id in self.initial_task_nodes:
            node_id = self.initial_task_nodes[task_id]
            res_man = self.node_resources[node_id]
            if res_man.can_run_task(task_spec['compute_demand'], task_spec['memory_demand'] * 1024, task_spec['model_required']):
                return node_id
            else:
                if self.log_manager.verbose:
                    self.log_manager.log_print(f"WARN: Pre-assigned node {node_id} for task {task_id} no longer suitable. Finding another.")
                # Fall through to general search, removing pre-assignment
                del self.initial_task_nodes[task_id]


        # Delegate to the current scheduler strategy
        # find_suitable_node in strategy might return a node_spec dict or just node_id
        # Let's standardize it to return node_spec dict
        _t0 = _time.process_time()
        suitable_node_spec = self.scheduler.find_suitable_node(scheduler_task_spec)
        _t1 = _time.process_time()
        self.decision_time_records.append((_t1 - _t0) * 1000.0)
        if suitable_node_spec:
            return suitable_node_spec['id']
        return None

    def start_task(self, task_id: str, node_id: str):
        task_spec = self.task_map[task_id]
        task_state = self.task_states[task_id]

        if task_state['status'] in ['running', 'completed', 'failed']:
            self.log_manager.log_print(f"[{self.env.now:.2f}] Task {task_id} already processed (status: {task_state['status']}). Skipping start.")
            return False

        if not self.node_resources[node_id].can_run_task(
            task_spec['compute_demand'], 
            task_spec['memory_demand'] * 1024, # GB to MB
            task_spec['model_required']
        ):
            self.log_manager.log_print(f"[{self.env.now:.2f}] Node {node_id} cannot run task {task_id} (resource check failed at start). Adding back to ready queue.")
            # 记录失败节点，供重试阶段避开
            try:
                if hasattr(self.scheduler, 'failed_node_history'):
                    self.scheduler.failed_node_history[task_id].add(node_id)
            except Exception:
                pass
            # 不再直接失败任务，而是将任务放回待调度队列
            self.ready_to_schedule_tasks.add(task_id)
            task_state['status'] = 'ready'  # 更改状态为ready而不是failed
            # 设置一个小延迟后再尝试（避免连续重试）
            self.env.process(self._retry_task_later(task_id, delay=1.0))
            return False

        task_state.update({
            'status': 'running',
            'start_time': self.env.now,
            'node_assigned': node_id
        })
        self.currently_running_tasks.add(task_id)
        self.node_running_tasks[node_id].add(task_id)  # OPT: set.add O(1)
        
        if not self.task_reverse_deps.get(task_id, []): # Is a source task
            if self.first_task_node is None: # Capture the node of the very first task started
                 self.first_task_node = node_id

        self.log_manager.update_sim_time(self.env.now)
        self.log_manager.log_task_lifecycle(task_id, '开始执行', {
            'node': node_id, 'start_time': self.env.now,
            'required_flops': task_spec['compute_demand'],
            'required_memory': task_spec['memory_demand'],
            'required_model': task_spec['model_required']
        })

        # Use scheduler's start_task method which might have strategy-specific logic
        # or calls node_scheduler.schedule_task
        # NodeScheduler calculates actual duration based on its own logic usually.
        # The task_spec for scheduler needs to be consistent with NodeTask in NodeScheduler
        # Use task_state['duration_hint'] which correctly stores the generated duration
        node_task_obj = {
            'task_id': task_id,
            'compute_flops': task_spec['compute_demand'], # This is total FLOPs, not GFLOPS/sec
            'memory_mb': task_spec['memory_demand'] * 1024,
            'model_name': task_spec['model_required'],
            'duration': task_state['duration_hint'] # Correctly use duration_hint from task_state
        }
        if self.log_manager.verbose:
            self.log_manager.log_print(f"DEBUG CM: Task {task_id} passing duration {node_task_obj['duration']:.2f} (from task_state['duration_hint']={task_state['duration_hint']:.2f}) to NodeScheduler.")
        if hasattr(self.scheduler, 'on_task_start'):
            self.scheduler.on_task_start(task_id, node_id, task_state)

        # The callback needs to be bound to this instance of ClusterManager
        bound_complete_callback = self.complete_task_callback  # OPT: direct method reference instead of lambda
        
        # Using node_scheduler directly for now, as strategy.start_task was mostly a pass-through
        # If strategies need to intercept start_task, this needs to be self.scheduler.start_task
        success = self.node_schedulers[node_id].schedule_task(node_task_obj, bound_complete_callback)
        
        if not success: # e.g. NodeScheduler's internal check failed (should be rare if can_run_task passed)
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] NodeScheduler failed to schedule task {task_id} on {node_id}.")
            # 记录失败节点，供重试阶段避开
            try:
                if hasattr(self.scheduler, 'failed_node_history'):
                    self.scheduler.failed_node_history[task_id].add(node_id)
            except Exception:
                pass
            self._fail_task(task_id, node_id, "NodeScheduler rejected task at start")
            # Revert state changes if start failed after state update
            self.currently_running_tasks.discard(task_id)
            self.node_running_tasks[node_id].discard(task_id)  # OPT: set.discard O(1)
            task_state['status'] = 'pending' # Or 'failed'
            return False
        
        if self.log_manager.verbose:
            self.log_manager.log_print(f"[{self.env.now:.2f}] Task {task_id} started on node {node_id}.")
        return True

    def _fail_task(self, task_id: str, node_id: Optional[str], reason: str):
        task_state = self.task_states[task_id]
        task_state.update({
            'status': 'failed',
            'completion_time': self.env.now, # Failure time
            'failure_reason': reason,
            'node_assigned': node_id
        })
        self.failed_tasks_details.append({
            'task_id': task_id, 'time': self.env.now, 'reason': reason, 'node': node_id
        })
        self.currently_running_tasks.discard(task_id)  # OPT: set.discard O(1)
        if node_id:
            self.node_running_tasks.get(node_id, set()).discard(task_id)  # OPT: set.discard O(1)

        self.log_manager.update_sim_time(self.env.now)
        self.log_manager.log_task_lifecycle(task_id, '任务失败', {
            'node': node_id, 'reason': reason, 'time': self.env.now
        })
        if self.log_manager.verbose:
            self.log_manager.log_print(f"[{self.env.now:.2f}] Task {task_id} FAILED on node {node_id}. Reason: {reason}")
        if hasattr(self.scheduler, 'on_task_failure'):
            self.scheduler.on_task_failure(task_id, node_id, task_state)
        
        # Consider if failed tasks should trigger successors with a "failed_dependency" state
        # For now, successors of failed tasks will simply not have their dependencies met.

    def complete_task_callback(self, task_id: str, node_id: str):
        # This is called by NodeScheduler when a task finishes execution on a node
        task_state = self.task_states[task_id]
        # task_spec = self.task_map[task_id] # Not used currently in this method

        task_state.update({
            'status': 'completed',
            'completion_time': self.env.now
        })
        self.completed_tasks.add(task_id)
        self.currently_running_tasks.discard(task_id)
        self.node_running_tasks.get(node_id, set()).discard(task_id)  # OPT: set.discard O(1)
        
        self.node_completion_times[node_id] = self.env.now

        self.log_manager.update_sim_time(self.env.now)
        self.log_manager.log_task_lifecycle(task_id, '完成执行', {
            'node': node_id, 'completion_time': self.env.now,
            'execution_time': self.env.now - task_state['start_time'] if task_state['start_time'] is not None else -1
        })
        if self.log_manager.verbose:
            self.log_manager.log_print(f"[{self.env.now:.2f}] Task {task_id} COMPLETED on node {node_id}.")

        # Handle data transfers for successors and schedule them
        self._process_task_completion(task_id, node_id)
        if hasattr(self.scheduler, 'on_task_completion'):
            self.scheduler.on_task_completion(task_id, node_id, task_state)

    def _process_task_completion(self, completed_task_id: str, completed_node_id: str):
        # This method is crucial for DAG progression.
        # For each successor of the completed_task_id:
        # 1. Check if all its dependencies are now met.
        # 2. If so, schedule data transfers from all (completed) dependencies to the successor's target node.
        # 3. Once all data transfers for the successor are complete, mark successor as 'ready' and schedule it.

        for successor_id in self.task_deps.get(completed_task_id, []):
            if self.task_states[successor_id]['status'] != 'pending': continue # Already processed or running/failed

            # OPT: Use pending_dep_count for O(1) check instead of iterating all deps
            self.pending_dep_count[successor_id] -= 1
            if self.pending_dep_count[successor_id] > 0:
                continue  # Still has pending dependencies

            # All deps completed (count reached 0)
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] All dependencies for task {successor_id} are met.")

            # Record first ready time for timeout tracking (before any scheduling attempt)
            if successor_id not in self.task_ready_time:
                self.task_ready_time[successor_id] = self.env.now

            # All direct dependencies are 'completed'. Now, schedule data transfers.
            # First, decide on the target_node for successor_id.
            # This might involve calling find_suitable_node if not already assigned.

            # If successor_id already has a node from initial_task_nodes or prior assignment
            target_node_id_for_successor = self.task_assignments.get(successor_id)
            if not target_node_id_for_successor:
                 target_node_id_for_successor = self.find_suitable_node(successor_id)
            
            if not target_node_id_for_successor:
                # 对带重试功能的策略，进入延迟重试而不是立即失败
                # All strategies: retry instead of immediately failing the successor.
                if self.log_manager.verbose:
                    self.log_manager.log_print(f"[{self.env.now:.2f}] No suitable node for successor {successor_id}. Scheduling retry.")
                self.task_states[successor_id]['status'] = 'ready'
                self.ready_to_schedule_tasks.add(successor_id)  # OPT: set.add O(1)
                # task_ready_time already set above before find_suitable_node
                retry_iv = 1.0
                if hasattr(self.scheduler, 'retry_config'):
                    retry_iv = self.scheduler.retry_config.get('retry_interval', 1.0)
                self.env.process(self._retry_task_later(successor_id, delay=retry_iv))
                continue

            self.task_assignments[successor_id] = target_node_id_for_successor
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] Task {successor_id} assigned to node {target_node_id_for_successor}.")

            # List of data transfer events to wait for
            required_transfers: List[simpy.Event] = []
            for dep_id in self.task_reverse_deps.get(successor_id, []):
                source_node_id = self.task_states[dep_id]['node_assigned']
                dep_task_spec = self.task_map[dep_id]
                # FIX: data_size in task_spec is already in MB, not bytes — removed erroneous /(1024*1024)
                data_size_mb = dep_task_spec.get('data_size', 0) 

                if source_node_id != target_node_id_for_successor and data_size_mb > 0:
                    if self.log_manager.verbose:
                        self.log_manager.log_print(f"[{self.env.now:.2f}] Scheduling data transfer: {data_size_mb:.2f}MB from {dep_id}@{source_node_id} to {successor_id}@{target_node_id_for_successor}")
                    event = self.network_simulator.transfer_data(
                        source_node_id, target_node_id_for_successor, data_size_mb, 
                        f"{dep_id}_to_{successor_id}" # Task_id for logging within network_sim
                    )
                    required_transfers.append(event)
            
            if required_transfers:
                # Wait for all these transfers, then schedule the successor
                self.env.process(self._wait_for_transfers_and_schedule(successor_id, target_node_id_for_successor, required_transfers))
            else:
                # No data transfers needed, schedule successor immediately
                if self.log_manager.verbose:
                    self.log_manager.log_print(f"[{self.env.now:.2f}] No data transfers needed for {successor_id}. Scheduling immediately.")
                self.start_task(successor_id, target_node_id_for_successor)

    def _wait_for_transfers_and_schedule(self, task_id_to_schedule: str, node_id: str, transfer_events: List[simpy.Event]):
        if not transfer_events: # Should not happen if called from path with transfers
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] _wait_for_transfers called with empty event list for {task_id_to_schedule}. Scheduling directly.")
            self.start_task(task_id_to_schedule, node_id)
            return

        all_transfers_event = simpy.AllOf(self.env, transfer_events)
        yield all_transfers_event
        
        # Check if any transfer event failed (simpy events don't "fail" but might timeout if delay is inf)
        # Here, we assume NetworkSimulator handles very long/infinite delays by logging and returning quickly.
        # The important part is that 'yield' completes.
        
        if self.log_manager.verbose:
            self.log_manager.log_print(f"[{self.env.now:.2f}] All data transfers for task {task_id_to_schedule} to node {node_id} complete.")
        self.start_task(task_id_to_schedule, node_id)


    def schedule_task(self, task_id: str):
        """Schedules a single task. Used for initial tasks or could be part of a more complex scheduling loop."""
        if self.simulation_start_time < 0: # First ever call to schedule_task
            self.simulation_start_time = self.env.now
            for res_man in self.node_resources.values():
                res_man.reset_simulation_time(self.simulation_start_time)
            # Log initial state only once
            self.log_manager.log_initial_state(self.nodes_spec, self.node_resources) 
            # Log creation for all tasks at the beginning of simulation
            self.log_manager.update_sim_time(self.env.now) # current time is simulation_start_time
            for t_id, t_state in self.task_states.items(): # Iterate task_states to get full list
                 t_spec = self.task_map[t_id] # Get spec for details
                 t_state['creation_time'] = self.env.now
                 self.log_manager.log_task_lifecycle(t_id,'任务创建',{
                    'required_flops': t_spec['compute_demand'],
                    'required_memory': t_spec['memory_demand'],
                    'required_model': t_spec['model_required'],
                    'original_id': t_state.get('original_id', t_id)
                 })


        task_state = self.task_states[task_id]
        if task_state['status'] != 'pending':
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] Task {task_id} not in pending state ({task_state['status']}). Cannot schedule via this method.")
            return False

        # Record first ready time for timeout tracking (must be set before any scheduling attempt)
        if task_id not in self.task_ready_time:
            self.task_ready_time[task_id] = self.env.now

        # Check dependencies for this task_id
        # This method assumes it's OK to schedule (e.g. source task or deps handled by caller)
        # For initial tasks, this is fine.

        # If node is pre-assigned via set_initial_task_node, task_assignments will have it.
        node_id = self.task_assignments.get(task_id)
        if not node_id: # If not pre-assigned, find one
            node_id = self.find_suitable_node(task_id)

        if node_id:
            self.task_assignments[task_id] = node_id # Record assignment
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] Scheduling task {task_id} on node {node_id}.")
            # At this point, data transfers for this task (if it's not a source)
            # should have already occurred or been scheduled by _process_task_completion.
            # If this is a source task, no incoming data transfers.
            self.start_task(task_id, node_id)
            return True
        else:
            # All strategies: retry instead of immediately failing the task.
            # Resources may become available soon; immediate failure causes
            # cascading failures in DAG successors, producing artificially low CR.
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] No suitable node found for task {task_id}. Scheduling retry.")
            self.task_states[task_id]['status'] = 'ready'
            self.ready_to_schedule_tasks.add(task_id)  # OPT: set.add O(1)
            # task_ready_time already set at top of schedule_task
            retry_iv = 1.0
            if hasattr(self.scheduler, 'retry_config'):
                retry_iv = self.scheduler.retry_config.get('retry_interval', 1.0)
            self.env.process(self._retry_task_later(task_id, delay=retry_iv))
            return False

    def initialize_simulation(self):
        """Initializes states and logs, called once before simulation starts or first task is scheduled."""
        if self.simulation_start_time < 0: # Ensures this runs only once
            self.simulation_start_time = self.env.now # Should be 0 at this point
            for res_man in self.node_resources.values():
                res_man.reset_simulation_time(self.simulation_start_time)
            self.log_manager.log_initial_state(self.nodes_spec, self.node_resources)
            
            self.log_manager.update_sim_time(self.env.now)
            for t_id, t_state in self.task_states.items(): # Iterate task_states to get full list
                t_spec = self.task_map[t_id] # Get spec for details
                t_state['creation_time'] = self.env.now
                self.log_manager.log_task_lifecycle(t_id, '任务创建', {
                    'required_flops': t_spec['compute_demand'],
                    'required_memory': t_spec['memory_demand'],
                    'required_model': t_spec['model_required'],
                    'original_id': t_state.get('original_id', t_id)
                })
        
        # Optional: Print initial topology sort or other info
        # topo_order = self.topological_sort()
        # self.log_manager.log_print(f"Task topological order: {topo_order}")

    def update_system_status(self, priority_queue_for_stats: List = []): # priority_queue might not be used if not driving via _simulation_process
        """Updates and logs current system status. Can be called periodically by a monitor process."""
        running_tasks_info_list = []
        for task_id_running in self.currently_running_tasks:
            state = self.task_states.get(task_id_running, {})
            running_tasks_info_list.append({
                'task_id': task_id_running,
                'model': state.get('required_model', ''),
                'node': state.get('node_assigned', '')
            })
        
        # Count failed tasks by reason directly from self.failed_tasks_details
        resource_failed_counts = defaultdict(int)
        for f_detail in self.failed_tasks_details:
            reason = f_detail['reason']
            if '内存' in reason or 'memory' in reason: resource_failed_counts['memory_exceeded'] += 1
            elif '模型' in reason or 'model' in reason: resource_failed_counts['model_unavailable'] += 1
            elif '计算' in reason or 'compute' in reason: resource_failed_counts['compute_exceeded'] += 1
            elif '节点' in reason or 'node' in reason : resource_failed_counts['node_unavailable'] +=1 # Generic node issue
            else: resource_failed_counts['other']+=1


        status_info = {
            'queued_tasks': len(priority_queue_for_stats), # This count might be inaccurate if not using main loop
            'running_tasks': len(self.currently_running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks_details), # Total failed tasks
            'running_tasks_info': running_tasks_info_list,
            'failed_tasks_detail': self.failed_tasks_details,
            'resource_failed_counts': dict(resource_failed_counts)
        }
        
        # Aggregate resource status from all nodes
        total_used_flops, total_used_memory = 0, 0
        total_flops_cap, total_memory_cap = 0, 0
        all_node_models = set()

        for node_id, res_man in self.node_resources.items():
            node_stat = res_man.get_resource_status(current_time=self.env.now)
            total_used_flops += node_stat['used_flops']
            total_used_memory += node_stat['used_memory']
            total_flops_cap += node_stat['total_flops']
            total_memory_cap += node_stat['total_memory']
            all_node_models.update(node_stat['available_models'])
        
        status_info['resource_status'] = {
            'used_flops': total_used_flops, 'total_flops': total_flops_cap,
            'used_memory': total_used_memory, 'total_memory': total_memory_cap,
            'available_models': list(all_node_models)
        }
        
        self.log_manager.update_sim_time(self.env.now)
        self.log_manager.log_system_status(status_info) # This method is in LogManager

    def print_summary(self):
        self.log_manager.log_print("\n" + "="*20 + " SIMULATION SUMMARY " + "="*20)
        total_tasks_defined = len(self.tasks_spec)
        num_completed = len(self.completed_tasks)
        num_failed = len(self.failed_tasks_details)
        
        # Calculate pending/running based on states, not just subtraction, to be more robust
        num_pending_or_running_actual = 0
        for task_id in self.task_states:
            if self.task_states[task_id]['status'] in ['pending', 'running', 'ready']: # 'ready' could be a transient state
                num_pending_or_running_actual +=1
        
        self.log_manager.log_print(f"Total tasks defined (from DAG spec): {total_tasks_defined}")
        self.log_manager.log_print(f"Completed tasks: {num_completed}")
        self.log_manager.log_print(f"Failed tasks: {num_failed}")
        # num_not_finished = total_tasks_defined - num_completed - num_failed
        self.log_manager.log_print(f"Tasks still pending/running/ready at sim end: {num_pending_or_running_actual}")

        # Count cascade-blocked tasks: tasks stuck in 'pending' because a
        # predecessor failed (they can never have dependencies met).
        cascaded_blocked = 0
        for task_id in self.task_states:
            state = self.task_states[task_id]['status']
            if state == 'pending':
                # Check if any predecessor failed
                for pred_id in self.task_reverse_deps.get(task_id, []):
                    if self.task_states.get(pred_id, {}).get('status') == 'failed':
                        cascaded_blocked += 1
                        break
        if cascaded_blocked > 0:
            self.log_manager.log_print(f"Cascade-blocked tasks (predecessor failed): {cascaded_blocked}")


        if num_completed > 0:
            total_exec_time = sum(self.task_states[tid]['completion_time'] - self.task_states[tid]['start_time'] 
                                  for tid in self.completed_tasks 
                                  if self.task_states[tid].get('start_time') is not None and self.task_states[tid].get('completion_time') is not None)
            self.log_manager.log_print(f"Sum of execution times for completed tasks: {total_exec_time:.2f}s")
            self.log_manager.log_print(f"Average execution time for completed tasks: {total_exec_time / num_completed:.2f}s")
            
            completion_times = [self.task_states[tid]['completion_time'] for tid in self.completed_tasks if self.task_states[tid].get('completion_time') is not None]
            makespan = max(completion_times) if completion_times else 0
            self.log_manager.log_print(f"Makespan (last task completion time): {makespan:.2f}s")
        else:
            self.log_manager.log_print("No tasks completed.")
            self.log_manager.log_print(f"Makespan (last task completion time): N/A")


        # Count actual data transfers made through network simulator
        # This requires NetworkSimulator to track them, or rely on logs if specific format
        # For now, just using the initiated count.
        num_transfers_made = self.network_simulator.get_completed_transfer_count() if hasattr(self.network_simulator, 'get_completed_transfer_count') else "N/A (tracker not impl)"
        self.log_manager.log_print(f"Total data transfers initiated by ClusterManager: {len(self.data_transfer_events)}")
        # self.log_manager.log_print(f"Total data transfers completed by NetworkSimulator: {num_transfers_made}")


        self.log_manager.log_print("\nNode Resource Utilization Summary (at sim end):")
        sim_duration_for_util = self.env.now - (self.simulation_start_time if self.simulation_start_time >=0 else 0)
        if sim_duration_for_util <= 0: sim_duration_for_util = self.env.now # if start time was 0 and end is 0
        self.log_manager.log_print(f"Total simulation duration considered for utilization: {sim_duration_for_util:.2f}s")

        for node_id, res_man in self.node_resources.items():
            status = res_man.get_resource_status(current_time=self.env.now) # Get final status
            self.log_manager.log_print(
                f"Node {node_id}: "
                f"Avg FLOPS: {status['avg_flops_percent']:.2f}% ({status['avg_used_flops']:.2f}/{status['total_flops']} GFLOPS), "
                f"Avg Memory: {status['avg_memory_percent']:.2f}% ({status['avg_used_memory']:.2f}/{status['total_memory']} MB)"
            )
        self.log_manager.log_print("="*58)
