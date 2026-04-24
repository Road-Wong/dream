# dream_proactive_scheduler.py

import math
import random
import statistics
import time
from typing import Dict, List, Optional, Tuple, Deque, NamedTuple, Any
from collections import defaultdict, deque
from strategies.scheduler_strategies import SchedulerStrategy

# T7-1: 计时装饰器用于性能监控
def timeit(func):
    """计时装饰器，记录函数执行时间到日志（仅超过阈值时记录）"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000  # 转为毫秒

        # 阈值设置：CPLS 100ms, OCAP 50ms
        threshold_map = {
            '_recursive_path_search': 100.0,  # CPLS 搜索
            '_calculate_opportunity_cost': 50.0,  # OCAP 计算
        }
        threshold = threshold_map.get(func.__name__, 10.0)  # 默认 10ms

        if elapsed >= threshold:
            self_obj = args[0] if args else None
            if hasattr(self_obj, 'log_manager') and self_obj.log_manager:
                self_obj.log_manager.log_print(f"[TIMING] {func.__name__} took {elapsed:.2f}ms (>={threshold}ms threshold)")
            else:
                # 如果没有 log_manager，直接打印
                print(f"[TIMING] {func.__name__} took {elapsed:.2f}ms (>={threshold}ms threshold)")
        return result
    return wrapper

# --- Data structure for resource reservations ---

class Reservation(NamedTuple):
    task_id: str              # Task ID for which the reservation is made
    node_id: str              # Node ID where the reservation is made
    start_time: float         # Reserved start time
    end_time: float           # Reserved end time
    dynamic_criticality: float # Priority of the reservation

class ReservationTable:
    def __init__(self):
        # Stores reservations per node: Dict[node_id, List[Reservation]]
        self.reservations = defaultdict(list)

    def add_reservation(self, node_id: str, reservation: Reservation):
        # Cleanup old reservations for this node
        now = reservation.start_time
        valid = []
        for res in self.reservations[node_id]:
            if res.end_time >= now - 1e-6:  # keep active and near-future ones
                valid.append(res)
        valid.append(reservation)
        self.reservations[node_id] = valid

    def get_overlaps(self, node_id: str, window_start: float, window_end: float) -> List[Reservation]:
        overlaps = []
        if node_id not in self.reservations:
            return overlaps
        # Cleanup stale before checking
        now = window_start
        self.reservations[node_id] = [r for r in self.reservations[node_id] if r.end_time >= now - 1e-6]
        for res in self.reservations[node_id]:
            if max(window_start, res.start_time) < min(window_end, res.end_time):
                overlaps.append(res)
        return overlaps

    def get_next_reservation(self, node_id: str, reference_time: float) -> Optional[Reservation]:
        """Return the earliest upcoming reservation on node `node_id` and prune stale ones."""
        if node_id not in self.reservations:
            return None

        next_reservation = None
        future_entries = []

        for res in self.reservations[node_id]:
            if res.end_time < reference_time - 1e-6:
                continue
            future_entries.append(res)
            if res.start_time >= reference_time - 1e-6:
                if next_reservation is None or res.start_time < next_reservation.start_time:
                    next_reservation = res

        self.reservations[node_id] = future_entries
        return next_reservation

    def get_stats(self, current_time: float) -> Dict[str, int]:
        """Return statistics about the reservation table."""
        total_reservations = 0
        active_reservations = 0
        expired_reservations = 0
        for node_id, res_list in self.reservations.items():
            total_reservations += len(res_list)
            for res in res_list:
                if res.end_time >= current_time - 1e-6:
                    active_reservations += 1
                else:
                    expired_reservations += 1
        return {
            "total_reservations": total_reservations,
            "active_reservations": active_reservations,
            "expired_reservations": expired_reservations,
            "nodes_with_reservations": len(self.reservations)
        }

# --- DREAM Proactive Scheduler Implementation ---

class DREAMProactiveScheduler(SchedulerStrategy):
    """
    DREAM (Dynamic, Ripple-Effect-Aware Meta-Scheduler) Proactive Implementation.

    Features:
    1. Dynamic Criticality: Prioritizes tasks based on runtime delays.
    2. Dual-Mode Scheduling: Uses CPLS for critical tasks and OCAP for non-critical tasks.
    3. CPLS (Critical Path Lookahead Scheduling): Plans and reserves resources for critical path segments.
    4. OCAP (Opportunity Cost-Aware Placement): Avoids conflicts with critical task reservations.
    """
    
    # P1-4 FIX: Network bandwidth aligned with paper Fig.1 topology.
    # (latency_ms, bandwidth_MB/s) per link type
    _NETWORK_PARAMS = {
        ('cloud', 'cloud'): (5, 9200),    # was 1000 — paper Fig.1: ~9200 MB/s
        ('edge', 'edge'): (10, 800),      # was 200  — paper Fig.1: ~800 MB/s
        ('end', 'end'): (2, 50),
        ('cloud', 'edge'): (20, 1500),    # was 150  — paper Fig.1: ~1500 MB/s
        ('edge', 'cloud'): (20, 1500),    # was 150  — symmetric
        ('edge', 'end'): (5, 80),
        ('end', 'edge'): (5, 80),
        'default': (100, 20)  # cloud-end (slowest)
    }
    
    # Default network parameters for averages
    _AVG_LATENCY_MS = 15.0
    _AVG_BANDWIDTH_MBPS = 100.0
    _LOAD_BALANCE_THRESHOLD = 0.8
    _OVERLOAD_PENALTY_FACTOR = 10.0

    def __init__(self, env, nodes, tasks, dependencies, node_resources, node_schedulers,
                 log_manager=None, get_node_current_status_fn=None, get_task_state_fn=None,
                 config: Dict[str, Any] = None, network_simulator=None):
        
        # Handle different task data structure formats before calling parent
        if isinstance(tasks, dict):
            tasks_list = list(tasks.values())
            self.task_map = tasks
        else:
            # Assume tasks is a list of task specs
            tasks_list = tasks
            self.task_map = {task['id']: task for task in tasks if isinstance(task, dict) and 'id' in task}
        
        # Handle different node data structure formats  
        if isinstance(nodes, dict):
            nodes_list = list(nodes.values())
        else:
            nodes_list = nodes
        
        # Convert dependencies format for parent class
        # Parent expects list of {'source': task_id, 'target': successor_id}
        # We have dict {task_id: [successor_ids]}
        dependencies_list = []
        if isinstance(dependencies, dict):
            for source_task, successor_list in dependencies.items():
                for target_task in successor_list:
                    dependencies_list.append({'source': source_task, 'target': target_task})
        else:
            dependencies_list = dependencies
            
        super().__init__(env, nodes_list, tasks_list, dependencies_list, node_resources, node_schedulers,
                        log_manager, get_node_current_status_fn, get_task_state_fn)

        # --- Configuration ---
        config = config or {}
        # CPLS parameters
        self.cpls_lookahead_depth = config.get("cpls_depth", 4)
        self.cpls_branching_factor = config.get("cpls_branching", 2)  # P1-3 FIX: 默认值改为 2（对齐论文最优配置）
        # P2-A FIX: 多路径 CPLS (ENH-01) 作为工程可扩展项，论文 Algorithm 1 以注释形式说明
        # 论文描述：核心贡献为单路径 CPLS，多路径为可扩展实现
        self.cpls_paths = config.get("cpls_paths", 1)  # P1-3 FIX: 默认值改为 1（对齐论文）
        
        # P1-3 FIX: 打印 CPLS 参数配置日志
        if self.log_manager:
            self.log_manager.log_print(
                f"[DREAM INIT] cpls_branching_factor={self.cpls_branching_factor}, "
                f"cpls_depth={self.cpls_lookahead_depth}, cpls_paths={self.cpls_paths}"
            )
        # P1-C FIX: 代码默认权重对齐论文值 (0.44, 0.27, 0.17, 0.12)
        # BEFORE: (0.38, 0.27, 0.17, 0.18) 与论文不一致
        # AFTER:  (0.44, 0.27, 0.17, 0.12) 与论文 Section V-B 一致
        self.w_immediate = config.get("w_immediate", 0.44)   # 论文遗传算法最优权重 w_imm
        self.w_ripple = config.get("w_ripple", 0.27)         # 论文 w_rip
        self.w_stability = config.get("w_stability", 0.17)    # 论文 w_stab
        self.w_opportunity = config.get("w_opportunity", 0.12)  # 论文 w_opp (非 0.18)
        # FIX: w_locality enabled (was 0.0) — co-locate dependent tasks to reduce cross-node transfers
        self.w_locality = config.get("w_locality", 0.10)
        self.eft_tolerance = config.get("eft_tolerance", 0.25)  # LB-3 FIX: expanded from 0.15 to 0.25 to allow more load-balanced candidates
        self.successor_mode = config.get("successor_mode", "max_rank")
        self.successor_top_k = max(1, int(config.get("successor_top_k", self.cpls_branching_factor)))
        self.retry_boost_factor = config.get("retry_boost_factor", 0.12)
        self.near_critical_margin = config.get("near_critical_margin", 0.08)
        self.reservation_alignment_bonus = config.get("reservation_alignment_bonus", 0.18)
        # Enhanced backfill/overlap controls
        self.backfill_bonus_factor = config.get("backfill_bonus_factor", 0.6)
        # F3 FIX: 基础乘子从 3.6 降至 1.5，并配置化
        # BEFORE: self.opportunity_overlap_escalation = config.get("overlap_escalation", 3.6)
        self.opportunity_overlap_escalation = config.get("opp_escalation_base", 1.5)
        # F3 FIX: 新增调节因子参数
        self.opp_dc_factor = config.get("opp_dc_factor", 0.5)        # DC 调节因子（从 1.2 降至 0.5）
        self.opp_load_factor = config.get("opp_load_factor", 0.8)    # 负载调节因子（从 1.5 降至 0.8）
        self.opp_overlap_factor = config.get("opp_overlap_factor", 0.5)  # 重叠调节因子（从 1.0 降至 0.5）
        self.opportunity_overlap_reservation_scale = config.get("overlap_reservation_scale", 1.35)
        # Node pressure weighting
        self.w_pressure = config.get("w_pressure", 0.18)
        self.pressure_queue_norm = config.get("pressure_queue_norm", 6)
        self.pressure_wait_horizon = config.get("pressure_wait_horizon", 8.0)
        # Enhanced critical path controls
        self.aggressive_critical_threshold = config.get("aggressive_critical_threshold", 0.85)
        self.reservation_priority_boost = config.get("reservation_priority_boost", 3.8)
        # Advanced optimization parameters
        # P1-2 FIX: high_load_threshold 0.72→0.80 — defer high-load weight
        # adjustment to avoid premature departure from normal scheduling behavior
        self.high_load_threshold = config.get("high_load_threshold", 0.80)
        # P1-2 FIX: extreme_load_threshold 0.85→0.92 — defer extreme-load
        # weight adjustment to preserve normal scheduling over wider load range
        self.extreme_load_threshold = config.get("extreme_load_threshold", 0.92)
        # P1-2 FIX: stability_boost_factor 1.5→0.8 — excessive stability boost
        # under extreme load causes tasks to over-concentrate on already-loaded nodes
        self.stability_boost_factor = config.get("stability_boost_factor", 0.8)
        self.preemption_threshold = config.get("preemption_threshold", 0.95)
        self.bottleneck_soft_limit = config.get("bottleneck_soft_limit", 0.9)
        self.bottleneck_hard_limit = config.get("bottleneck_hard_limit", 0.95)
        self.bottleneck_recovery_limit = config.get("bottleneck_recovery_limit", 0.78)
        # Predictive optimization
        self.prediction_window = config.get("prediction_window", 6)
        self.prediction_buffer_max = config.get("prediction_buffer_max", max(6, self.prediction_window * 2))
        self.load_history_window = config.get("load_history_window", max(12, self.prediction_window * 3))
        self.trend_smoothing_alpha = config.get("trend_smoothing_alpha", 0.35)
        self.adaptive_learning_rate = config.get("adaptive_learning_rate", 0.1)
        self.workload_prediction_enabled = config.get("workload_prediction", True)
        # Dynamic Criticality parameters
        # P1-1 FIX: criticality_threshold_ratio 0.7→0.85 — raise DC threshold
        # to reduce CPLS over-invocation under load
        self.criticality_threshold_ratio = config.get("criticality_threshold_ratio", 0.85)
        self.criticality_alpha = config.get("criticality_alpha", 2.0)
        self.use_mean_std_threshold = config.get("use_mean_std_threshold", True)
        self.cpls_soft_reservation = bool(config.get("cpls_soft_reservation", False))
        self.enable_ocap_normalization = config.get("enable_ocap_normalization", True)
        # Reservation guard buffer: when cpls_soft_reservation is enabled,
        # extend reservation windows by this fraction of execution time so that
        # non-critical tasks are more likely to be steered away from reserved nodes.
        self.reservation_guard_factor = config.get("reservation_guard_factor", 0.5)
        self.use_network_tt = config.get("use_network_tt", True)
        self.use_priority_in_dc = bool(config.get("use_priority_in_dc", False))
        self.use_retry_boost_in_dc = bool(config.get("use_retry_boost_in_dc", False))
        self.network_pair_sample_ratio = max(0.0, min(1.0, config.get("network_pair_sample_ratio", 1.0)))
        self.avg_task_duration_fallback = config.get("avg_task_duration", 5.0) # Fallback value
        # P1-1 FIX: max_wait_penalty 5.0→2.0 — reduce DC dynamism penalty so
        # fewer tasks get inflated criticality under high load
        self.max_wait_penalty = config.get("max_wait_penalty", 2.0)
        self.lambda_penalty = config.get("lambda_penalty", 3.0)  # P1-1 FIX: C_opp 中的常数惩罚因子（论文 λ）; raised from 1.0 to make reservations impactful
        self.priority_weights = config.get("priority_weights", {
            "critical": 1.5,
            "high": 1.2,
            "medium": 1.0,
            "normal": 0.9,
            "low": 0.7
        })

        # P2-A FIX: Bandit 学习机制在 P0-B 修复后关闭，不作为论文核心描述
        # 理由：参数复杂度已引发 R2/R4 质疑，避免分散核心贡献焦点
        self.bandit_enabled = bool(config.get("bandit_enabled", False))  # False = 与论文一致
        self.bandit_epsilon = max(0.0, min(1.0, config.get("bandit_epsilon", 0.08)))
        user_bandit_arms = config.get("bandit_arms")
        if isinstance(user_bandit_arms, list):
            self.bandit_arms = [dict(arm) for arm in user_bandit_arms if isinstance(arm, dict)]
        else:
            self.bandit_arms = []
        if not self.bandit_arms:
            self.bandit_arms = [
                {"w_immediate": 1.0, "w_opportunity": 1.0, "w_ripple": 1.0, "w_stability": 1.0, "w_locality": 1.0},
                {"w_immediate": 1.15, "w_opportunity": 0.85, "w_ripple": 0.95, "w_stability": 1.05, "w_locality": 1.0},
                {"w_immediate": 0.9, "w_opportunity": 1.1, "w_ripple": 1.05, "w_stability": 1.0, "w_locality": 1.1}
            ]
        self.bandit_reward_decay = max(0.0, min(1.0, config.get("bandit_reward_decay", 0.05)))
        self.bandit_counts = [1.0 for _ in self.bandit_arms]
        self.bandit_values = [0.0 for _ in self.bandit_arms]
        self.bandit_active_arm = None
        self.bandit_last_context = None
        self.network_simulator = network_simulator or config.get("network_simulator")
        self._avg_pair_comm_cache = {}
        # Debug configuration
        self.debug_cpls = config.get("debug_cpls", False)  # Enable CPLS debug logging

        # --- State and Precomputation ---
        self.reservation_table = ReservationTable()
        self._cpls_hints = {}  # task_id -> List[{"node_id": str, "start_time": float, "end_time": float, "dc": float}]
        self._cpls_hints_cleanup_counter = 0  # periodic cleanup counter
        self.upward_ranks = self._compute_upward_ranks()
        self.max_upward_rank = max(self.upward_ranks.values()) if self.upward_ranks else 1.0
        # Precompute global upward rank statistics for stable DC threshold
        _rank_vals = list(self.upward_ranks.values()) if self.upward_ranks else [1.0]
        _rank_mean = sum(_rank_vals) / len(_rank_vals)
        _rank_var = sum((x - _rank_mean) ** 2 for x in _rank_vals) / len(_rank_vals)
        self._global_rank_mean = _rank_mean
        self._global_rank_std = _rank_var ** 0.5
        self.task_state_cache = {} # Caches for intermediate calculations
        
        # Advanced state tracking
        self.system_load_history = []
        self.performance_metrics = {'successful_placements': 0, 'failed_placements': 0}
        self.adaptive_threshold_cache = {}
        self.bottleneck_nodes = set()  # Track problematic nodes
        self.task_retry_counts = defaultdict(int)
        self._current_task_retry_count = 0
        
        # --- Scheduling statistics counters (paper-aligned reporting) ---
        self._cpls_call_count: int = 0
        self._ocap_call_count: int = 0
        self._critical_task_count: int = 0
        self._total_dc_eval_count: int = 0
        self._reservation_overlap_count: int = 0  # times a reservation overlap was detected in OCAP
        self._cpls_decision_time_ms: List[float] = []  # per-call CPLS CPU time (ms)

        # Defect B fix: sliding window of DC values for dynamic threshold
        self._dc_history = deque(maxlen=200)
        self.ocap_decision_counts: Dict[str, int] = {'pure_eft': 0, 'multi_obj': 0}  # OCAP decision type counters

        # (Removed: CPLS retry cap — if a task is critical, it always goes through CPLS per the paper's algorithm)
        
        # Predictive analytics
        self.task_completion_predictions = {}
        self.node_availability_predictions = {}
        self.critical_path_predictions = {}
        self.workload_trend_buffer = []
        
        # --- Performance optimization caches ---
        self._node_lookup_cache = {node['id']: node for node in self.nodes_spec}
        self._feasible_nodes_cache = {}  # Cache feasible nodes per task requirements
        self._comm_time_cache = {}  # Cache communication times
        self._exec_time_cache = {}  # Cache execution times

        if log_manager and log_manager.verbose:
            log_manager.log_print("Initialized DREAMProactiveScheduler with performance optimizations.")

    def get_scheduling_stats(self) -> Dict[str, float]:
        """Returns scheduling statistics for paper reporting."""
        total = max(self._total_dc_eval_count, 1)
        total_ocap_decisions = max(self.ocap_decision_counts['pure_eft'] + self.ocap_decision_counts['multi_obj'], 1)
        # Reservation table statistics
        reservation_stats = self.reservation_table.get_stats(self.env.now) if hasattr(self, 'env') and self.env else {}
        cpls_dt_mean = (sum(self._cpls_decision_time_ms) / len(self._cpls_decision_time_ms)
                        if self._cpls_decision_time_ms else 0.0)
        return {
            "cpls_calls": self._cpls_call_count,
            "cpls_decision_time_mean_ms": cpls_dt_mean,
            "ocap_calls": self._ocap_call_count,
            "critical_task_count": self._critical_task_count,
            "total_dc_evals": self._total_dc_eval_count,
            "critical_ratio": self._critical_task_count / total,
            "reservation_overlap_count": self._reservation_overlap_count,
            "ocap_pure_eft_ratio": self.ocap_decision_counts['pure_eft'] / total_ocap_decisions,
            "ocap_multi_obj_ratio": self.ocap_decision_counts['multi_obj'] / total_ocap_decisions,
            **reservation_stats,
        }
    # ... (rest of the code remains the same)
    # --- Core Dispatcher ---

    def find_suitable_node(self, task_spec: Dict) -> Optional[Dict]:
        """
        Main scheduling entry point.
        Dispatches to CPLS or OCAP based on dynamic criticality.
        """
        task_id = task_spec.get('id', 'unknown')
        attempts = self.task_retry_counts.get(task_id, 0)
        self._current_task_retry_count = attempts

        # Invalidate per-round caches to prevent stale data
        self._feasible_nodes_cache.clear()
        self._exec_time_cache.clear()
        self._comm_time_cache.clear()
        
        try:
            # 1. Calculate Dynamic Criticality
            dynamic_criticality, is_critical = self._calculate_dynamic_criticality(task_spec)
            task_spec['dynamic_criticality'] = dynamic_criticality # Augment task spec for internal use

            best_node = None
            
            # 2. Critical Task: Use CPLS with fallback to OCAP
            if is_critical:
                # Critical tasks always go through CPLS (per paper's algorithm)
                if self.log_manager and self.log_manager.verbose:
                    self.log_manager.log_print(f"Task {task_id}: CRITICAL (DC={dynamic_criticality:.2f}). Running CPLS...")
                    
                    try:
                        decision, reservations = self._run_cpls(task_spec)
                        if decision:
                            if self.log_manager and self.log_manager.verbose:
                                self.log_manager.log_print(f"CPLS successful for task {task_id}")
                            # Don't reset CPLS counter even on success - just return
                            # Store CPLS reservations in ReservationTable (paper design).
                            # Soft reservations still affect T_avail via FindAvailTime and
                            # are penalized through C_opp in OCAP (Eq. 27: ω·DC·λ_penalty).
                            if reservations:
                                for res in reservations:
                                    self.reservation_table.add_reservation(res.node_id, res)
                                if self.debug_cpls:
                                    print(f"[CPLS_RESERVATIONS] task {task_id}: stored {len(reservations)} reservations")
                            # CRITICAL FIX: Also create a reservation for the current critical task
                            # itself. This ensures that when other tasks are scheduled while this
                            # critical task is still pending/running, they are steered away from
                            # the same node, preserving Cloud resources for the critical path.
                            # This is the primary mechanism by which α affects scheduling quality:
                            #   low α → more critical tasks → more reservations → more nodes "blocked"
                            #   → non-critical tasks forced to use sub-optimal nodes → different makespan
                            eft_now, st_now, et_now = self._calculate_immediate_cost(task_spec, decision)
                            if eft_now < float('inf') and st_now > 0:
                                current_dc = task_spec.get('dynamic_criticality', dynamic_criticality)
                                # When cpls_soft_reservation is enabled, pad the
                                # reservation window so it blocks a larger time
                                # range, making OCAP more likely to avoid this node.
                                res_end = et_now
                                if self.cpls_soft_reservation:
                                    exec_dur = et_now - st_now
                                    res_end = et_now + exec_dur * self.reservation_guard_factor
                                self_reservation = Reservation(
                                    task_id=task_id,
                                    node_id=decision['id'],
                                    start_time=st_now,
                                    end_time=res_end,
                                    dynamic_criticality=current_dc
                                )
                                self.reservation_table.add_reservation(decision['id'], self_reservation)
                                if self.debug_cpls:
                                    print(f"[CPLS_SELF_RES] task {task_id} on {decision['id']}: [{st_now:.1f}, {et_now:.1f}]")
                            self.task_retry_counts[task_id] = 0
                            self._current_task_retry_count = 0
                            return decision
                        else:
                            if self.log_manager and self.log_manager.verbose:
                                self.log_manager.log_print(f"CPLS failed for critical task {task_id}, falling back to OCAP")
                    except Exception as e:
                        if self.log_manager:
                            self.log_manager.log_print(f"CPLS error for task {task_id}: {e}")
                        best_node = None
            
            # 3. Use OCAP (either for non-critical tasks or CPLS fallback)
            if self.log_manager and self.log_manager.verbose:
                task_type = "non-critical" if not is_critical else "critical (CPLS fallback)"
                self.log_manager.log_print(f"Task {task_id}: {task_type} (DC={dynamic_criticality:.2f}). Running OCAP...")
            
            try:
                best_node = self._run_ocap(task_spec)
                if best_node:
                    if self.log_manager and self.log_manager.verbose:
                        self.log_manager.log_print(f"OCAP successful for task {task_id}")
                    self.task_retry_counts[task_id] = 0
                    self._current_task_retry_count = 0
                    return best_node
                else:
                    if self.log_manager and self.log_manager.verbose:
                        self.log_manager.log_print(f"OCAP failed for task {task_id}, using greedy fallback")
            except Exception as e:
                if self.log_manager and self.log_manager.verbose:
                    self.log_manager.log_print(f"OCAP exception for task {task_id}: {str(e)}, using greedy fallback")
            
            # 4. All CPLS/OCAP methods failed — use greedy fallback
            best_node = self._greedy_fallback(task_spec)
            if best_node:
                if self.log_manager and self.log_manager.verbose:
                    self.log_manager.log_print(f"Greedy fallback placed task {task_id} on node {best_node['id']}")
            self.task_retry_counts[task_id] = attempts + 1
            self._current_task_retry_count = 0
            return best_node
            
        except Exception as e:
            # Catch-all exception handler — try greedy fallback
            if self.log_manager and self.log_manager.verbose:
                self.log_manager.log_print(f"Critical error scheduling task {task_id}: {str(e)}")
            self.task_retry_counts[task_id] = attempts + 1
            self._current_task_retry_count = 0
            return self._greedy_fallback(task_spec)

    # --- Dynamic Criticality Calculation ---

    def _calculate_dynamic_criticality(self, task_spec: Dict) -> Tuple[float, bool]:
        if not task_spec or 'id' not in task_spec:
            return 0.0, False
        task_id = task_spec['id']
        static_rank = max(self.upward_ranks.get(task_id, 0.0), 0.0)
        current_time = self.env.now if self.env else 0.0
        tstate = self.get_task_state(task_id, {}) if hasattr(self, 'get_task_state') else {}
        preds = self.task_reverse_deps.get(task_id, [])
        if preds:
            t_ready = 0.0
            for p in preds:
                ps = self.get_task_state(p, {})
                ct = ps.get('completion_time', 0.0)
                t_ready = max(t_ready, ct)
        else:
            t_ready = 0.0
        wait_penalty = 0.0
        atd = self._compute_atd()
        # Debug ATD and wait time
        if self.log_manager and (self.log_manager.verbose or self.debug_cpls):
            self.log_manager.log_print(f"DC_DEBUG: task {task_id}, current_time={current_time:.2f}, t_ready={t_ready:.2f}, wait={current_time - t_ready:.2f}, atd={atd:.6f}")
        if self.debug_cpls:
            print(f"[DC_DEBUG] task {task_id}, current_time={current_time:.2f}, t_ready={t_ready:.2f}, wait={current_time - t_ready:.2f}, atd={atd:.6f}")
        if atd > 0.0:
            wait_penalty = max(0.0, current_time - t_ready) / atd
            # Cap wait penalty to prevent explosion
            # BEFORE: max_wait_penalty = 2.0 硬编码
            if self.max_wait_penalty > 0:
                wait_penalty = min(self.max_wait_penalty, wait_penalty)
        dynamic_criticality = static_rank * (1.0 + wait_penalty)
        # Record DC value in sliding window before threshold check
        self._dc_history.append(dynamic_criticality)
        if self.use_priority_in_dc:
            priority = self.task_map.get(task_id, {}).get('priority', tstate.get('priority', 'normal'))
            priority_weight = self.priority_weights.get(priority, self.priority_weights.get('medium', 1.0))
            dynamic_criticality *= priority_weight
        if self.use_retry_boost_in_dc:
            retry_boost = 1.0 + self._current_task_retry_count * self.retry_boost_factor
            dynamic_criticality *= retry_boost
        if self.use_mean_std_threshold:
            # Defect B fix: compute threshold from DC values WITH wait_penalty
            # so threshold scales with actual system-wide wait conditions
            ready_dcs = self._get_ready_queue_dynamic_criticalities()
            if ready_dcs and len(ready_dcs) >= 2:
                dc_mean = sum(ready_dcs) / len(ready_dcs)
                dc_var = sum((x - dc_mean) ** 2 for x in ready_dcs) / len(ready_dcs)
                dc_std = dc_var ** 0.5
                threshold = dc_mean + self.criticality_alpha * dc_std
            else:
                # Fallback: global rank stats with conservative capping
                threshold = min(
                    self._global_rank_mean + self.criticality_alpha * self._global_rank_std,
                    self.max_upward_rank * self.criticality_threshold_ratio
                )
        else:
            # Legacy path: should not be reached when use_mean_std_threshold=True
            if getattr(self, 'log_manager', None) and getattr(self.log_manager, 'verbose', False):
                self.log_manager.log_print(
                    "WARN [DC]: Legacy ratio-threshold branch triggered. "
                    "Check use_mean_std_threshold config."
                )
            max_rank = max(self.max_upward_rank, 1.0)
            threshold = max_rank * self.criticality_threshold_ratio
        is_critical = dynamic_criticality >= threshold
        self._total_dc_eval_count += 1
        if is_critical:
            self._critical_task_count += 1

        # Debug logging
        if self.log_manager and (self.log_manager.verbose or self.debug_cpls):
            self.log_manager.log_print(f"DC: task {task_id}, static_rank={static_rank:.2f}, wait_penalty={wait_penalty:.2f}, DC={dynamic_criticality:.2f}, threshold={threshold:.2f}, is_critical={is_critical}")
        # Always print to console for debugging if critical
        # Debug logging removed: print(f"[DEBUG] CRITICAL TASK: {task_id}, DC={dynamic_criticality:.2f}, threshold={threshold:.2f}")

        return dynamic_criticality, is_critical

    def _compute_atd(self) -> float:
        """计算平均任务时长（论文 Eq.20：mean(W_i) / mean(C_j)，单位：秒）

        # BEFORE (错误):
        # w_list = [task['compute_demand'] for ...]  # ReqC_i (GFLOPS)，误用为 W_i
        # mw = mean(w_list)  # 单位 GFLOPS，无量纲比值
        # return mw / mc     # 无量纲，≈ 0.03，导致 wait_penalty 58ms 即饱和

        # AFTER (修正):
        # W_i = ReqC_i × ET_i (GFLOP)，ATD = mean(W_i) / mean(C_j) (秒)
        """
        try:
            # P0-B FIX: 计算 W_i = ReqC_i × duration (GFLOP)，而非直接使用 ReqC_i (GFLOPS)
            w_list = []
            for t in self.task_map:
                spec = self.task_map[t]
                req_c = spec.get('compute_demand', 1.0)   # ReqC_i, GFLOPS
                dur = spec.get('duration', self.avg_task_duration_fallback)  # ET_i, s
                w_list.append(req_c * dur)                 # W_i = ReqC_i × ET_i, GFLOP
            c_list = [n.get('compute_power', 0.0) for n in self.nodes_spec]
            mw = sum(w_list) / max(len(w_list), 1)         # mean(W_i), GFLOP
            mc = sum(c_list) / max(len(c_list), 1)         # mean(C_j), GFLOPS
            if mc > 0.0:
                result = mw / mc                           # 单位：GFLOP/GFLOPS = 秒
                # Debug ATD calculation
                if self.log_manager and (self.log_manager.verbose or self.debug_cpls):
                    self.log_manager.log_print(f"ATD_DEBUG: mw={mw:.2f} GFLOP, mc={mc:.2f} GFLOPS, atd={result:.4f}s, len(w)={len(w_list)}, len(c)={len(c_list)}")
                if self.debug_cpls:
                    print(f"[ATD_DEBUG] mw={mw:.2f} GFLOP, mc={mc:.2f} GFLOPS, atd={result:.4f}s, len(w)={len(w_list)}, len(c)={len(c_list)}")
                return max(result, 1e-6)  # 确保返回正值，避免除零
            return self.avg_task_duration_fallback
        except Exception:
            return self.avg_task_duration_fallback

    def _get_ready_queue_dynamic_criticalities(self) -> List[float]:
        """Defect B fix: compute DC values (with wait_penalty) for all ready tasks."""
        current_time = self.env.now if self.env else 0.0
        atd = self._compute_atd()
        dcs = []
        for task_id in self.task_map:
            tstate = self.get_task_state(task_id, {}) if hasattr(self, 'get_task_state') else {}
            status = tstate.get('status', '')
            # F1 FIX: 严格仅包含就绪队列（不含 pending）
            # BEFORE: if status not in ('ready', 'pending'): # 包含 'ready' 和 'pending'
            if status != 'ready':
                continue
            static_rank = self.upward_ranks.get(task_id, 0.0)
            if static_rank <= 0:
                continue
            # Compute wait_penalty (same logic as _calculate_dynamic_criticality)
            t_ready = 0.0
            for p in self.task_reverse_deps.get(task_id, []):
                ps = self.get_task_state(p, {})
                ct = ps.get('completion_time') or 0.0
                t_ready = max(t_ready, ct)
            wait_penalty = 0.0
            if atd > 0.0:
                wait_penalty = max(0.0, current_time - t_ready) / atd
                if self.max_wait_penalty > 0:
                    wait_penalty = min(self.max_wait_penalty, wait_penalty)
            dc = static_rank * (1.0 + wait_penalty)
            dcs.append(dc)
        return dcs

    # --- Algorithm 1: CPLS (Critical Path Lookahead Scheduling) ---

    def _run_cpls(self, task_spec: Dict) -> Tuple[Optional[Dict], List[Reservation]]:
        _cpls_t0 = time.process_time()
        try:
            self._cpls_call_count += 1
            task_id = task_spec.get('id')
            init_dc = task_spec.get('dynamic_criticality', self.upward_ranks.get(task_id, 0.0))

            # Debug logging
            if self.log_manager and (self.log_manager.verbose or self.debug_cpls):
                self.log_manager.log_print(f"CPLS called for task {task_id} (DC={init_dc:.3f})")
            if self.debug_cpls:
                print(f"[DEBUG] CPLS called for task {task_id} (DC={init_dc:.3f})")

            # CPLS always uses full lookahead depth regardless of hint count.
            # The α trade-off comes purely from computational overhead:
            #   low α → more CPLS calls → higher total decision time → O(D·K^D) cost
            #   (not from artificially degrading CPLS quality)
            effective_depth = self.cpls_lookahead_depth

            # ENH-01: Multi-Path CPLS — explore top K_PATHS entry successors
            K_PATHS = getattr(self, 'cpls_paths', 2)  # Default: 2 paths (conservative)
            successors = self.task_deps.get(task_id, [])
            if successors:
                # Rank successors by score (same logic as _identify_critical_path_segment)
                successor_scores = []
                for succ_id in successors:
                    comm_time = self._estimate_avg_comm_time(task_id, succ_id)
                    succ_rank = self.upward_ranks.get(succ_id, 0)
                    score = comm_time + succ_rank
                    successor_scores.append((score, succ_id))
                successor_scores.sort(reverse=True)
                top_successors = [sid for _, sid in successor_scores[:K_PATHS]]
            else:
                top_successors = []

            global_best_plan = None
            global_best_makespan = float('inf')
            # CPLS uses mode="cpls" for diversified candidate selection:
            # considers load balance + successor comm, not just EFT-first.
            # This enables CPLS to choose different nodes than OCAP,
            # making reservations actually constrain OCAP's later decisions.
            cpls_k = max(1, int(self.cpls_branching_factor))
            candidates = self._get_top_k_feasible_nodes(task_spec, cpls_k, mode="cpls")
            if self.debug_cpls:
                print(f"[CPLS_DEBUG] task {task_id}: candidates={len(candidates)}, top_successors={len(top_successors)}")

            if len(top_successors) <= 1:
                # Single path or no successors — use original single-path logic
                path = self._identify_critical_path_segment(task_id, effective_depth)
                if not path:
                    path = [task_id]
                if self.debug_cpls:
                    print(f"[CPLS_DEBUG] task {task_id}: path length={len(path)}, path={path}")
                for node_spec in candidates:
                    eft, st, et = self._calculate_immediate_cost(task_spec, node_spec)
                    plan = self._recursive_path_search(path[1:], [(task_id, node_spec)], eft, init_dc)
                    if plan and plan["makespan"] < global_best_makespan:
                        global_best_makespan = plan["makespan"]
                        global_best_plan = plan
            else:
                # Multi-path: build a separate path segment from each top successor
                for entry_succ in top_successors:
                    # Build path: [task_id, entry_succ, ...]
                    sub_path = self._identify_critical_path_segment(entry_succ, effective_depth - 1)
                    path = [task_id] + sub_path

                    for node_spec in candidates:
                        eft, st, et = self._calculate_immediate_cost(task_spec, node_spec)
                        plan = self._recursive_path_search(path[1:], [(task_id, node_spec)], eft, init_dc)
                        if plan and plan["makespan"] < global_best_makespan:
                            global_best_makespan = plan["makespan"]
                            global_best_plan = plan

            if global_best_plan:
                if self.log_manager and (self.log_manager.verbose or self.debug_cpls):
                    self.log_manager.log_print(f"CPLS found plan for task {task_id} with makespan {global_best_makespan:.2f}, reservations: {len(global_best_plan.get('reservations', []))}")
                return global_best_plan["placement"][0][1], global_best_plan["reservations"]
            else:
                if self.log_manager and (self.log_manager.verbose or self.debug_cpls):
                    self.log_manager.log_print(f"CPLS failed to find plan for task {task_id}")
                if self.debug_cpls:
                    print(f"[CPLS_FAIL] task {task_id}: no plan found")
            return None, []
        except Exception as e:
            if self.debug_cpls:
                import traceback
                print(f"[CPLS_EXCEPTION] task {task_id}: {e}")
                traceback.print_exc()
            return None, []
        finally:
            self._cpls_decision_time_ms.append((time.process_time() - _cpls_t0) * 1000.0)

    @timeit
    def _recursive_path_search(self, path_segment: List[str], current_placement: List[Tuple[str, Dict]], current_finish_time: float, init_dc: float) -> Optional[Dict]:
        """Recursive function to explore path placements."""
        # Base Case: Path segment planning complete
        if not path_segment:
            reservations = []
            # Defect A fix: propagate cumulative EFT along the path instead of anchoring on env.now
            cumulative_eft = current_finish_time  # start from the actual EFT of the first scheduled task

            for i, (task_id, node_spec) in enumerate(current_placement[1:]):
                planned_task_spec = self.task_map[task_id]
                # Use per-task DC for reservations (not init_dc), so C_opp differentiates
                # reservations by actual task priority (paper Eq.27: r.DC)
                dc = self.upward_ranks.get(task_id, init_dc)
                # Apply wait penalty if task is already waiting
                tstate = self.get_task_state(task_id, {}) if hasattr(self, 'get_task_state') else {}
                t_ready = 0.0
                for p in self.task_reverse_deps.get(task_id, []):
                    ps = self.get_task_state(p, {})
                    ct = ps.get('completion_time') or 0.0
                    t_ready = max(t_ready, ct)
                atd = self._compute_atd()
                if atd > 0.0:
                    wp = max(0.0, (self.env.now if self.env else 0.0) - t_ready) / atd
                    if self.max_wait_penalty > 0:
                        wp = min(self.max_wait_penalty, wp)
                    dc = dc * (1.0 + wp)

                # Calculate communication from previous task in the placement chain
                prev_task_id, prev_node_spec = current_placement[i]
                prev_task_spec = self.task_map[prev_task_id]
                comm_time = self._calculate_comm_time(prev_task_spec, prev_node_spec, node_spec)

                res_start = cumulative_eft + comm_time  # Defect A: cumulative EFT, not env.now
                exec_time = self._get_execution_time(planned_task_spec, node_spec)
                res_end = res_start + exec_time
                cumulative_eft = res_end  # propagate to next task

                # When cpls_soft_reservation is enabled, pad successor reservation
                # windows so they block a larger time range for OCAP avoidance.
                padded_end = res_end
                if self.cpls_soft_reservation:
                    exec_dur = res_end - res_start
                    padded_end = res_end + exec_dur * self.reservation_guard_factor
                reservations.append(Reservation(task_id, node_spec['id'], res_start, padded_end, dc))

            return {"makespan": current_finish_time, "placement": current_placement, "reservations": reservations}

        # Recursive Step: Plan for the next task in segment
        next_task_id = path_segment[0]
        next_task_spec = self.task_map[next_task_id]
        predecessor_task_id, predecessor_node = current_placement[-1]

        best_subtree_plan = {"makespan": float('inf')}
        cpls_k = max(1, int(self.cpls_branching_factor))
        candidate_nodes = self._get_top_k_feasible_nodes(next_task_spec, cpls_k, mode="cpls")

        for next_node_spec in candidate_nodes:
            # Calculate start time for next_task at next_node_spec (include resource wait)
            comm_time = self._calculate_comm_time(self.task_map[predecessor_task_id], predecessor_node, next_node_spec)
            tentative_start = max(self.env.now, current_finish_time + comm_time)
            res_man_next = self.node_resources[next_node_spec['id']]
            if res_man_next.has_available_resources(
                next_task_spec.get('compute_demand', 1.0),
                next_task_spec.get('memory_demand', 1.0) * 1024
            ):
                resource_ready_time_next = self.env.now
            else:
                resource_ready_time_next = self.env.now + self._estimate_wait_time(res_man_next, next_task_spec)
            
            # P0-3 FIX: 补全 off-path 前驱的 DRT 估计（论文 Algorithm 2 第 5 行 + Eq.24-25）
            path_task_ids = set(task_id for task_id, _ in current_placement)
            off_path_drt = 0.0
            for pred_id in self.task_reverse_deps.get(next_task_id, []):
                if pred_id not in path_task_ids:
                    pred_spec = self.task_map.get(pred_id, {})
                    pred_et = pred_spec.get('duration', self.avg_task_duration_fallback)
                    estimated_ct = self.env.now + pred_et    # 论文 Eq.24 乐观估计
                    # 计算平均通信时间：从该前驱到目标节点
                    pred_nodes = self._get_feasible_nodes(pred_spec)
                    est_tt = 0.0
                    if pred_nodes:
                        # 取前几个可行节点计算平均 TT
                        sample_nodes = pred_nodes[:3] if len(pred_nodes) > 3 else pred_nodes
                        total_tt = 0.0
                        for pn in sample_nodes:
                            total_tt += self._calculate_comm_time(pred_spec, pn, next_node_spec)
                        est_tt = total_tt / len(sample_nodes)
                    off_path_drt = max(off_path_drt, estimated_ct + est_tt)
            start_time = max(tentative_start, off_path_drt, resource_ready_time_next)
            exec_time = self._get_execution_time(next_task_spec, next_node_spec)
            finish_time = start_time + exec_time

            # No hint conflict penalty: CPLS should always produce its best plan.
            # The trade-off is purely computational: more CPLS calls = more time,
            # not worse quality per call.
            finish_time_adjusted = finish_time

            # Recurse
            recursive_result = self._recursive_path_search(
                path_segment=path_segment[1:],
                current_placement=current_placement + [(next_task_id, next_node_spec)],
                current_finish_time=finish_time_adjusted,
                init_dc=init_dc
            )

            if recursive_result and recursive_result["makespan"] < best_subtree_plan["makespan"]:
                best_subtree_plan = recursive_result
        
        return best_subtree_plan if best_subtree_plan["makespan"] != float('inf') else None

    # --- Algorithm 2: OCAP (Opportunity Cost-Aware Placement) ---

    def _run_ocap(self, task_spec: Dict) -> Optional[Dict]:
        try:
            self._ocap_call_count += 1
            if self.enable_ocap_normalization:
                return self._run_ocap_normalized(task_spec)
            best_node, min_total_cost = None, float('inf')
            feasible_nodes = self._get_feasible_nodes(task_spec)
            task_id = task_spec.get('id', 'unknown')

            if not feasible_nodes:
                if self.log_manager and self.log_manager.verbose:
                    self.log_manager.log_print(f"No feasible nodes found for task {task_id} in OCAP")
                self.performance_metrics['failed_placements'] += 1
                self._update_bandit(0.0)
                return None

            dc = task_spec.get('dynamic_criticality', 0.0)
            max_rank = max(self.max_upward_rank, 1.0)
            dc_norm = max(0.0, min(1.0, dc / max_rank))

            system_load = self._calculate_system_load()
            workload_trend = self._predict_workload_trend()

            trend_adjustment = 0.0
            if workload_trend > 0.05:
                trend_adjustment = min(0.2, workload_trend * 2.0)
            elif workload_trend < -0.05:
                trend_adjustment = max(-0.15, workload_trend * 1.5)

            effective_load = system_load + trend_adjustment

            w_immediate_eff = self.w_immediate * (1.0 + 0.4 * dc_norm)
            w_opportunity_eff = self.w_opportunity * (1.0 + 0.3 * dc_norm)
            w_ripple_eff = self.w_ripple * (1.0 - 0.2 * dc_norm)
            # LB-2 FIX: Do NOT reduce w_stability for critical tasks
            w_stability_eff = self.w_stability * (1.0 + 0.1 * dc_norm)

            if effective_load > self.high_load_threshold:
                load_adjustment = min(1.0, (system_load - self.high_load_threshold) / 0.1)
                w_stability_eff *= (1.0 + 0.4 * load_adjustment)
                # P1-2 FIX: High-load weight direction correction.
                # Original code decreased w_opp and increased w_imm under high load,
                # causing tasks to concentrate on strongest nodes, ignoring reservation
                # hints. Corrected: boost w_opp (protect reservations) and reduce
                # w_imm (avoid over-concentration on fastest nodes).
                w_opportunity_eff *= (1.0 + 0.4 * load_adjustment)
                w_immediate_eff *= (1.0 - 0.2 * load_adjustment)

                if system_load > self.extreme_load_threshold:
                    extreme_adjustment = min(1.0, (system_load - self.extreme_load_threshold) / 0.1)
                    w_stability_eff *= (1.0 + self.stability_boost_factor * extreme_adjustment)
                    # P1-2 FIX: Extreme load — same direction correction as high-load.
                    # Reduce w_ripple and w_imm, keep w_opp elevated to maintain
                    # reservation-aware placement even under extreme contention.
                    w_ripple_eff *= (1.0 - 0.2 * extreme_adjustment)
                    w_immediate_eff *= (1.0 - 0.1 * extreme_adjustment)   # mild reduction
                    w_opportunity_eff *= (1.0 + 0.1 * extreme_adjustment)  # mild boost

            w_ripple_eff = max(w_ripple_eff, 0.35 * self.w_ripple)
            w_opportunity_eff = max(w_opportunity_eff, 0.6 * self.w_opportunity)

            bandit_modifiers = self._select_bandit_modifiers(dc_norm, effective_load)
            if bandit_modifiers:
                w_immediate_eff *= bandit_modifiers.get("w_immediate", 1.0)
                w_opportunity_eff *= bandit_modifiers.get("w_opportunity", 1.0)
                w_ripple_eff *= bandit_modifiers.get("w_ripple", 1.0)
                w_stability_eff *= bandit_modifiers.get("w_stability", 1.0)
            locality_multiplier = bandit_modifiers.get("w_locality", 1.0) if bandit_modifiers else 1.0

            res_for_task = self._get_reservations_for_task(task_id)
            reserved_node_ids = {r.node_id for r in res_for_task} if res_for_task else set()

            for node_spec in feasible_nodes:
                try:
                    c_immediate, start_time, end_time = self._calculate_immediate_cost(task_spec, node_spec)
                    if c_immediate == float('inf'):
                        continue

                    c_ripple = self._calculate_ripple_cost(task_spec, node_spec)
                    c_stability = self._calculate_stability_cost(task_spec, node_spec)
                    c_opportunity = self._calculate_opportunity_cost(node_spec, start_time, end_time)

                    res_align_pen = 0.0
                    if reserved_node_ids:
                        if node_spec['id'] in reserved_node_ids:
                            closest = min(res_for_task, key=lambda r: abs(start_time - r.start_time))
                            res_align_pen = 0.05 * abs(start_time - closest.start_time)
                        else:
                            avg_res_span = sum((r.end_time - r.start_time) for r in res_for_task) / len(res_for_task)
                            res_align_pen = (0.8 + dc_norm + max(0.0, effective_load - 0.6)) * max(0.0, avg_res_span)

                    backfill_bonus = 0.0
                    next_res = self._get_next_reservation_for_node(node_spec['id'], self.env.now)
                    if next_res:
                        if end_time <= next_res.start_time:
                            slack = max(0.0, next_res.start_time - end_time)
                            slack_bonus = self.backfill_bonus_factor * min(slack, c_immediate)
                            if dc_norm > 0.7 and slack <= 0.5 * (next_res.end_time - next_res.start_time):
                                slack_bonus *= 1.5
                            backfill_bonus = slack_bonus
                        else:
                            overlap_ratio = max(0.0, min(end_time, next_res.end_time) - max(start_time, next_res.start_time)) / max(1e-6, next_res.end_time - next_res.start_time)
                            # F3 FIX: escalation 乘子降级
                            # BEFORE: base=3.6, dc=1.2, load=1.5, overlap=1.0
                            # AFTER:  base=1.5, dc=0.5, load=0.8, overlap=0.5
                            escalation = (self.opportunity_overlap_escalation *  # 1.5（默认）
                                          (1.0 + self.opp_dc_factor * next_res.dynamic_criticality / max_rank) *  # DC 因子降低
                                          (1.0 + max(0.0, effective_load - 0.6) * self.opp_load_factor) *  # 负载因子降低
                                          (1.0 + self.opp_overlap_factor * overlap_ratio))  # 重叠因子降低
                            c_opportunity *= escalation

                    c_locality = self._calculate_locality_cost(task_spec, node_spec)
                    locality_weight = self.w_locality * (1.0 + 0.8 * dc_norm + max(0.0, effective_load - 0.6)) * locality_multiplier
                    alignment_bonus = 0.0
                    if reserved_node_ids and node_spec['id'] in reserved_node_ids:
                        alignment_bonus = self.reservation_alignment_bonus * (1.0 + dc_norm)

                    base_cost = (w_immediate_eff * c_immediate +
                                w_ripple_eff * c_ripple +
                                w_stability_eff * c_stability +
                                w_opportunity_eff * (c_opportunity + 2.0 * res_align_pen) +
                                locality_weight * c_locality)

                    # LB-1 FIX: Overload hard constraint in non-normalized path
                    try:
                        node_status = self.get_node_current_status(node_spec['id'])
                        cpu_u = node_status.get('used_flops', 0) / node_spec['compute_power'] if node_spec['compute_power'] > 0 else 0.0
                        mem_u = node_status.get('used_memory', 0) / (node_spec['memory'] * 1024) if node_spec['memory'] > 0 else 0.0
                        node_u = max(cpu_u, mem_u)
                    except Exception:
                        node_u = 0.5
                    if node_u > self._LOAD_BALANCE_THRESHOLD:
                        excess = node_u - self._LOAD_BALANCE_THRESHOLD
                        base_cost += self._OVERLOAD_PENALTY_FACTOR * (excess ** 2)

                    if system_load > self.extreme_load_threshold and dc_norm > self.preemption_threshold:
                        emergency_weight = min(0.8, (system_load - self.extreme_load_threshold) * 8.0)
                        total_cost = (emergency_weight * c_immediate +
                                     (1.0 - emergency_weight) * base_cost - backfill_bonus - alignment_bonus)
                    else:
                        total_cost = base_cost - backfill_bonus - alignment_bonus

                    if not (0 <= total_cost < float('inf')):
                        if self.log_manager and self.log_manager.verbose:
                            self.log_manager.log_print(f"Invalid cost calculated for node {node_spec.get('id', 'unknown')}: {total_cost}")
                        continue

                    if total_cost < min_total_cost:
                        min_total_cost, best_node = total_cost, node_spec

                except Exception as e:
                    if self.log_manager and self.log_manager.verbose:
                        self.log_manager.log_print(f"Error calculating cost for node {node_spec.get('id', 'unknown')}: {str(e)}")
                    continue

            if best_node:
                self.performance_metrics['successful_placements'] += 1
                total_attempts = (self.performance_metrics['successful_placements'] +
                                  self.performance_metrics['failed_placements'])
                success_rate = self.performance_metrics['successful_placements'] / max(total_attempts, 1)

                reward = self._compute_bandit_reward(min_total_cost, dc_norm, effective_load)
                self._update_bandit(reward)

                if total_attempts > 50 and success_rate > 0.9:
                    self.high_load_threshold = max(0.75, self.high_load_threshold - self.adaptive_learning_rate * 0.01)
                    self.extreme_load_threshold = max(0.85, self.extreme_load_threshold - self.adaptive_learning_rate * 0.01)
                elif success_rate < 0.7:
                    self.high_load_threshold = min(0.85, self.high_load_threshold + self.adaptive_learning_rate * 0.01)
                    self.extreme_load_threshold = min(0.95, self.extreme_load_threshold + self.adaptive_learning_rate * 0.01)

                if self.log_manager and self.log_manager.verbose:
                    self.log_manager.log_print(
                        f"OCAP selected node {best_node.get('id', 'unknown')} for task {task_id} "
                        f"with cost {min_total_cost:.4f} (success rate: {success_rate:.3f})"
                    )
                return best_node

            self.performance_metrics['failed_placements'] += 1
            self._update_bandit(0.0)
            if self.log_manager and self.log_manager.verbose:
                self.log_manager.log_print(f"OCAP failed to find suitable node for task {task_id}")
            return None

        except Exception as e:
            self.performance_metrics['failed_placements'] += 1
            self._update_bandit(0.0)
            if self.log_manager and self.log_manager.verbose:
                self.log_manager.log_print(f"OCAP failed for task {task_spec.get('id', 'unknown')}: {str(e)}")
            return None

    def _run_ocap_normalized(self, task_spec: Dict) -> Optional[Dict]:
        """OCAP with adaptive weights and bandit learning (paper-aligned design).
        
        Paper Eq. 28: total cost = w^T · Ĉ where Ĉ = (Ĉ_imm, Ĉ_rip, Ĉ_stab, Ĉ_opp).
        C_opp is computed via Eq. 27: Σ ω(v_i,p_j,r) · r.DC · λ_penalty.
        Reservations from CPLS affect T_avail and C_opp only (unidirectional).
        """
        try:
            self._ocap_call_count += 1

            candidates = self._get_feasible_nodes(task_spec)
            if not candidates:
                self.performance_metrics['failed_placements'] += 1
                return None

            task_id = task_spec.get('id', 'unknown')
            # === Stage 1: Compute costs for all candidates ===
            imm_list = []
            rip_list = []
            stab_list = []
            opp_list = []
            triplets = []
            for node in candidates:
                eft, st, et = self._calculate_immediate_cost(task_spec, node)
                if eft == float('inf'):
                    continue
                imm = eft
                rip_total = 0.0
                succs = self.task_deps.get(task_id, [])
                if succs:
                    for sid in succs:
                        succ_spec = self.task_map.get(sid, {})
                        succ_nodes = self._get_feasible_nodes(succ_spec)
                        min_tt = float('inf')
                        for sn in succ_nodes:
                            tt = self._calculate_comm_time(task_spec, node, sn)
                            if tt < min_tt:
                                min_tt = tt
                        if min_tt == float('inf'):
                            min_tt = 0.0
                        rip_total += imm + min_tt  # P0-1 FIX: EFT 对每个后继独立贡献（论文 Eq.29）
                rip = rip_total
                stab = self._calculate_stability_cost(task_spec, node)
                opp = self._calculate_opportunity_cost(node, st, et)
                imm_list.append(imm)
                rip_list.append(rip)
                stab_list.append(stab)
                opp_list.append(opp)
                triplets.append((node, imm, rip, stab, opp, st, et))
            if not triplets:
                # All feasible nodes are currently busy (c_immediate=inf).
                # Return None — the greedy fallback in find_suitable_node will handle this.
                self.performance_metrics['failed_placements'] += 1
                return None

            # === Stage 2: EFT Contender Filtering ===
            eft_min = min(imm_list)
            EFT_TOLERANCE = self.eft_tolerance
            contender_indices = [
                i for i, eft in enumerate(imm_list)
                if eft <= eft_min * (1 + EFT_TOLERANCE)
            ]

            # Defect E fix: even single contender, compute and log C_opp
            if hasattr(self, 'ocap_decision_counts'):
                if len(contender_indices) == 1:
                    self.ocap_decision_counts['pure_eft'] += 1
                    # Still record the opp cost for diagnostics
                    single_idx = contender_indices[0]
                    t = triplets[single_idx]
                    if self.log_manager and self.log_manager.verbose:
                        self.log_manager.log_print(
                            f"OCAP single contender node {t[0].get('id')}, opp={t[4]:.4f}")
                else:
                    self.ocap_decision_counts['multi_obj'] += 1

            # Extract contender data
            c_triplets = [triplets[i] for i in contender_indices]
            c_imm = [t[1] for t in c_triplets]
            c_rip = [t[2] for t in c_triplets]
            c_stab = [t[3] for t in c_triplets]
            c_opp = [t[4] for t in c_triplets]

            # P1-A FIX: OCAP 归一化引入绝对差异保护阈值
            # BEFORE: 仅处理 vmax == vmin，微小差异 (<1%) 被放大至全区间，C_stab 主导决策
            # AFTER: 当 rng < epsilon_abs 时视为统计无差异，返回全 0（论文 Eq.26 的合理扩展）
            def norm_list(vals, epsilon_abs=None):
                vmin = min(vals)
                vmax = max(vals)
                rng = vmax - vmin
                # 当绝对差异 < epsilon 且相对差异极小时，视为统计无差异
                if epsilon_abs is not None and rng < epsilon_abs:
                    return [0.0] * len(vals)  # 同等优秀，C_imm 不干扰次级决策
                if rng < 1e-9:
                    return [0.0] * len(vals)
                return [(v - vmin) / rng for v in vals]

            # epsilon_abs = 平均任务时长的 5%，作为统计有意义的阈值
            epsilon_threshold = self.avg_task_duration_fallback * 0.05
            n_imm = norm_list(c_imm, epsilon_abs=epsilon_threshold)
            n_rip = norm_list(c_rip)
            n_stab = norm_list(c_stab)
            n_opp = norm_list(c_opp)

            # === Defect F fix: Adaptive weight mechanisms (ported from old _run_ocap) ===
            dc = task_spec.get('dynamic_criticality', 0.0)
            max_rank = max(self.max_upward_rank, 1.0)
            dc_norm = max(0.0, min(1.0, dc / max_rank))
            system_load = self._calculate_system_load()
            workload_trend = self._predict_workload_trend()

            trend_adjustment = 0.0
            if workload_trend > 0.05:
                trend_adjustment = min(0.2, workload_trend * 2.0)
            elif workload_trend < -0.05:
                trend_adjustment = max(-0.15, workload_trend * 1.5)

            effective_load = system_load + trend_adjustment

            # 1. DC adaptive weights
            w_imm_eff = self.w_immediate * (1.0 + 0.4 * dc_norm)
            w_opp_eff = self.w_opportunity * (1.0 + 0.3 * dc_norm)
            w_rip_eff = self.w_ripple * (1.0 - 0.2 * dc_norm)
            # LB-2 FIX: Do NOT reduce w_stability for critical tasks.
            # BEFORE: w_stab_eff = w_stability * (1.0 - 0.3 * dc_norm)
            # This caused critical tasks to ignore load balance entirely,
            # concentrating on the fastest node. Keep stability weight
            # intact so even critical tasks respect global load distribution.
            w_stab_eff = self.w_stability * (1.0 + 0.1 * dc_norm)  # slight boost for critical tasks

            # 2. P1-2 FIX: high-load adaptive — boost w_opp (protect reservations),
            # reduce w_imm (avoid over-concentration on fastest nodes)
            if effective_load > self.high_load_threshold:
                load_adj = min(1.0, (system_load - self.high_load_threshold) / 0.1)
                w_stab_eff *= (1.0 + 0.4 * load_adj)
                w_opp_eff *= (1.0 + 0.4 * load_adj)         # P1-2: boost opp under high load
                w_imm_eff *= (1.0 - 0.2 * load_adj)          # P1-2: reduce imm, avoid concentration

                if system_load > self.extreme_load_threshold:
                    extreme_adj = min(1.0, (system_load - self.extreme_load_threshold) / 0.1)
                    w_stab_eff *= (1.0 + self.stability_boost_factor * extreme_adj)
                    # Under extreme load, moderate adjustments — don't over-correct
                    w_rip_eff *= (1.0 - 0.2 * extreme_adj)
                    w_imm_eff *= (1.0 - 0.1 * extreme_adj)    # mild reduction
                    w_opp_eff *= (1.0 + 0.1 * extreme_adj)    # mild boost

            # Floor values to prevent total collapse
            w_rip_eff = max(w_rip_eff, 0.35 * self.w_ripple)
            w_opp_eff = max(w_opp_eff, 0.6 * self.w_opportunity)

            # 3. Bandit modifiers
            bandit_modifiers = self._select_bandit_modifiers(dc_norm, effective_load)
            locality_multiplier = bandit_modifiers.get("w_locality", 1.0) if bandit_modifiers else 1.0
            if bandit_modifiers:
                w_imm_eff *= bandit_modifiers.get("w_immediate", 1.0)
                w_opp_eff *= bandit_modifiers.get("w_opportunity", 1.0)
                w_rip_eff *= bandit_modifiers.get("w_ripple", 1.0)
                w_stab_eff *= bandit_modifiers.get("w_stability", 1.0)

            # === Stage 3: Multi-objective scoring (paper-aligned) ===
            # Paper design: C_opp already computed in Stage 1 via _calculate_opportunity_cost,
            # which implements Eq. 27: C_opp = Σ ω(v_i,p_j,r) · r.DC · λ_penalty.
            # No separate hint/reservation alignment bonus or penalty.
            # The reservation table affects T_avail (FindAvailTime) and C_opp only.
            best = None
            best_cost = float('inf')

            # Compute absolute C_opp range for unnormalized reservation penalty
            opp_abs_min = min(c_opp) if c_opp else 0
            opp_abs_max = max(c_opp) if c_opp else 0
            opp_abs_range = opp_abs_max - opp_abs_min

            # LB-1 FIX: Pre-compute node utilization for overload hard constraint
            # Activate previously dead _LOAD_BALANCE_THRESHOLD / _OVERLOAD_PENALTY_FACTOR
            node_util_map = {}
            for n_spec in self.nodes_spec:
                try:
                    status = self.get_node_current_status(n_spec['id'])
                    cpu_util = status.get('used_flops', 0) / n_spec['compute_power'] if n_spec['compute_power'] > 0 else 0.0
                    mem_util = status.get('used_memory', 0) / (n_spec['memory'] * 1024) if n_spec['memory'] > 0 else 0.0
                    node_util_map[n_spec['id']] = max(cpu_util, mem_util)
                except Exception:
                    node_util_map[n_spec['id']] = 0.5

            for i, (node, imm, rip, stab, opp, st, et) in enumerate(c_triplets):
                # Locality cost
                locality_weight = self.w_locality * (1.0 + 0.8 * dc_norm + max(0.0, effective_load - 0.6)) * locality_multiplier
                c_locality = self._calculate_locality_cost(task_spec, node)

                # LB-1 FIX: Overload hard constraint — penalize nodes above threshold
                overload_penalty = 0.0
                node_util = node_util_map.get(node['id'], 0.5)
                if node_util > self._LOAD_BALANCE_THRESHOLD:
                    excess = node_util - self._LOAD_BALANCE_THRESHOLD
                    overload_penalty = self._OVERLOAD_PENALTY_FACTOR * (excess ** 2)

                # Total cost = w^T · Ĉ (paper Eq. 28)
                # C_opp already includes reservation overlap penalties from Eq. 27
                normalized_cost = (w_imm_eff * n_imm[i] +
                             w_rip_eff * n_rip[i] +
                             w_stab_eff * n_stab[i] +
                             w_opp_eff * n_opp[i] +
                             locality_weight * c_locality +
                             overload_penalty)

                # Reservation impact: add unnormalized C_opp penalty.
                # Without this, normalization dilutes the reservation signal:
                # if only 1 of 7 nodes has overlap, n_opp ∈ {0,1}, contributing
                # at most w_opp=0.12 — negligible vs w_imm=0.44.
                # The unnormalized term ensures reservations meaningfully steer OCAP.
                # Scale factor: use avg_task_duration as reference so the absolute
                # C_opp (in time units) is comparable to normalized scores (~1.0).
                opp_abs_penalty = 0.0
                if opp > 1e-9 and opp_abs_range > 1e-9:
                    # Scale absolute opp to be comparable with other normalized costs
                    # Use the ratio of opp to avg_task_duration as a dimensionless measure
                    opp_scaled = opp / self.avg_task_duration_fallback
                    # When cpls_soft_reservation is enabled, use reservation_priority_boost
                    # as the scaling factor so that reservations meaningfully steer OCAP
                    # away from reserved nodes. Without soft reservation mode, use the
                    # original w_opp_eff weight (backward compatible).
                    if self.cpls_soft_reservation:
                        opp_abs_penalty = self.reservation_priority_boost * opp_scaled
                    else:
                        opp_abs_penalty = w_opp_eff * opp_scaled

                total = normalized_cost + opp_abs_penalty

                if total < best_cost:
                    best_cost = total
                    best = node

            # === Defect J fix: Bandit learning loop and threshold adaptation ===
            if best is not None:
                self.performance_metrics['successful_placements'] += 1

                reward = self._compute_bandit_reward(best_cost, dc_norm, effective_load)
                self._update_bandit(reward)

                # Adaptive threshold adjustment
                total_attempts = (self.performance_metrics['successful_placements'] +
                                  self.performance_metrics['failed_placements'])
                success_rate = self.performance_metrics['successful_placements'] / max(total_attempts, 1)
                if total_attempts > 50 and success_rate > 0.9:
                    self.high_load_threshold = max(0.75, self.high_load_threshold - self.adaptive_learning_rate * 0.01)
                    self.extreme_load_threshold = max(0.85, self.extreme_load_threshold - self.adaptive_learning_rate * 0.01)
                elif success_rate < 0.7:
                    self.high_load_threshold = min(0.85, self.high_load_threshold + self.adaptive_learning_rate * 0.01)
                    self.extreme_load_threshold = min(0.95, self.extreme_load_threshold + self.adaptive_learning_rate * 0.01)

                if self.log_manager and self.log_manager.verbose:
                    self.log_manager.log_print(
                        f"OCAP-norm selected node {best.get('id', 'unknown')} for task {task_id} "
                        f"with cost {best_cost:.4f}, w_imm={w_imm_eff:.3f}, w_opp={w_opp_eff:.3f}, "
                        f"w_rip={w_rip_eff:.3f}, w_stab={w_stab_eff:.3f}, load={system_load:.3f}")
            else:
                self.performance_metrics['failed_placements'] += 1

            return best
        except Exception:
            return None

    @timeit
    def _calculate_opportunity_cost(self, node_spec: Dict, start_time: float, end_time: float) -> float:
        """Calculates penalty for overlapping with existing critical reservations."""
        cost = 0.0
        overlapping_reservations = self.reservation_table.get_overlaps(node_spec['id'], start_time, end_time)
        if overlapping_reservations:
            self._reservation_overlap_count += 1
        for res in overlapping_reservations:
            overlap_start = max(start_time, res.start_time)
            overlap_end = min(end_time, res.end_time)
            overlap_dur = max(0.0, overlap_end - overlap_start)
            res_dur = max(1e-6, res.end_time - res.start_time)
            weight = overlap_dur / res_dur
            # P1-1 FIX: 恢复常数 λ_penalty，移除非线性重叠因子
            # BEFORE: cost += res.dynamic_criticality * (1.0 + overlap_dur) * weight
            cost += weight * res.dynamic_criticality * self.lambda_penalty
        return cost

    # --- Cost Calculation Helpers (Shared Logic) ---

    def _calculate_immediate_cost(self, task_spec: Dict, node_spec: Dict) -> Tuple[float, float, float]:
        """Calculates EFT and returns (EFT, start_time, end_time).
        
        Implements FindAvailTime from the paper: scans the resource timeline
        including soft reservations to find the earliest available window.
        Soft reservations from CPLS delay T_avail (paper Section IV-B).
        """
        res_man = self.node_resources[node_spec['id']]
        exceeds, _ = res_man.exceeds_system_limits(
            task_spec.get('compute_demand', 1.0),
            task_spec.get('memory_demand', 1.0) * 1024,
            task_spec.get('model_required', 'default')
        )
        if exceeds:
            return float('inf'), 0.0, float('inf')
        data_ready_time = self._calculate_data_ready_time(task_spec, node_spec['id'])
        if res_man.has_available_resources(
            task_spec.get('compute_demand', 1.0),
            task_spec.get('memory_demand', 1.0) * 1024
        ):
            resource_ready_time = self.env.now
        else:
            resource_ready_time = self.env.now + self._estimate_wait_time(res_man, task_spec)
        start_time = max(self.env.now, data_ready_time, resource_ready_time)
        exec_time = self._get_execution_time(task_spec, node_spec)
        finish_time = start_time + exec_time
        
        # Paper: FindAvailTime considers soft reservations.
        # If the planned [start_time, finish_time] overlaps with a reservation
        # on this node, push start_time past the reservation to avoid conflict.
        # This implements the paper's statement that "soft reservations affect
        # the computation of T_avail for subsequent tasks."
        # 
        # ENHANCED: After pushing past one reservation, check again for further
        # overlaps (cascade). This ensures that if multiple reservations block
        # the node, the full delay is captured — making the Cloud node's EFT
        # reflect the actual wait, potentially making Edge/End nodes competitive.
        node_id = node_spec['id']
        max_iterations = 5  # prevent infinite loop
        for _ in range(max_iterations):
            overlaps = self.reservation_table.get_overlaps(node_id, start_time, finish_time)
            if not overlaps:
                break
            # Push start_time past the latest-ending overlapping reservation
            latest_end = max(r.end_time for r in overlaps)
            if latest_end <= start_time:
                break
            start_time = latest_end
            finish_time = start_time + exec_time
        
        return finish_time, start_time, finish_time

    def _calculate_ripple_cost(self, task_spec: Dict, node_spec: Dict) -> float:
        """Ripple cost per paper Eq.(12): min_{p_k in P_feasible(succ)} TT(p_j, p_k, Data_i,s).

        P0-2 FIX: Replaced avg_comm_time with min_comm_time over successor's feasible
        nodes, preserving topology heterogeneity. Removed reservation_conflict_penalty
        which duplicated C_opp's role and violated the paper's formula.
        """
        ripple_cost = 0.0
        successors = self.task_deps.get(task_spec['id'], [])
        if not successors:
            return 0.0

        total_weight = 0.0
        for succ_id in successors:
            succ_task_spec = self.task_map[succ_id]
            succ_rank = self.upward_ranks.get(succ_id, 0)
            weight = 1.0 + (succ_rank / max(self.max_upward_rank, 1.0))

            # P0-2 FIX: Use min_{p_k} TT(p_j, p_k, Data) per paper Eq.(12)
            # instead of avg_comm_time which erased topology heterogeneity.
            succ_feasible = self._get_feasible_nodes(succ_task_spec)
            if succ_feasible:
                min_comm_time = min(
                    self._calculate_comm_time(task_spec, node_spec, succ_node)
                    for succ_node in succ_feasible
                )
            else:
                min_comm_time = self._calculate_avg_comm_to_successors(task_spec, succ_task_spec, node_spec)

            ripple_cost += weight * min_comm_time
            total_weight += weight

        return ripple_cost / total_weight if total_weight > 0 else 0.0

    def _calculate_stability_cost(self, task_spec: Dict, node_spec: Dict) -> float:
        """
        Calculates C_stab (stability cost) using Gini coefficient of hypothetical utilization.
        Implements the formula from the paper: Gini = sum_a sum_b |u'_a - u'_b| / (2 * M * sum_k u'_k)
        Where u'_k = (used_flops_k + I(k==j)*W_i) / (C_k * T_eval)
        """
        import numpy as np

        try:
            # Step 1: Get all nodes and their current used flops
            all_nodes = self.nodes_spec
            M = len(all_nodes)
            if M <= 1:
                return 0.0  # No balance needed with single node

            # Step 2: Compute T_eval = max(T_now, EFT(v_i, p_j))
            eft, st, et = self._calculate_immediate_cost(task_spec, node_spec)
            T_eval = max(self.env.now if self.env else 0.0, eft)
            if T_eval <= 0:
                T_eval = 1.0  # Avoid division by zero

            # Step 3: Compute hypothetical utilization U'_k for all nodes
            # BEFORE: task_w = task_spec.get('compute_demand', 1.0)  # 实为 ReqC_i，量纲错误
            # AFTER: W_i = ReqC_i × ET_i (论文 Eq.31)
            req_c = task_spec.get('compute_demand', 1.0)
            et_val = task_spec.get('duration', self.avg_task_duration_fallback)
            task_w = req_c * et_val  # P0-2 FIX: W_i = ReqC_i × ET_i，GFLOP
            U_prime = []

            for k_node in all_nodes:
                status_k = self.get_node_current_status(k_node['id'])
                used_flops_k = status_k['used_flops']

                if k_node['id'] == node_spec['id']:
                    # Add this task's FLOPs to the target node
                    used_flops_k_hypothetical = used_flops_k + task_w
                else:
                    used_flops_k_hypothetical = used_flops_k

                # Compute u'_k = Used FLOPs / (Node capacity * T_eval)
                C_k = k_node['compute_power']  # GFLOPS
                if C_k <= 0:
                    u_prime_k = 1.0
                else:
                    u_prime_k = used_flops_k_hypothetical / (C_k * T_eval)
                U_prime.append(u_prime_k)

            U_prime = np.array(U_prime)

            # Step 4: Compute Gini coefficient
            sum_abs_diff = np.sum(np.abs(U_prime[:, None] - U_prime[None, :]))
            sum_u = np.sum(U_prime)

            if sum_u <= 0:
                gini = 0.0
            else:
                gini = sum_abs_diff / (2 * M * sum_u)

            # P2-A FIX: Hotspot Penalty (ENH-02) 作为 C_stab 的增强项
            # 论文对齐：在 Eq.30 (C_stab) 后补充说明此增强项用于密集关键任务场景 (R2 要求)
            # 核心仍为 Gini 系数，hotspot penalty 为工程鲁棒性增强
            mean_u = np.mean(U_prime)
            std_u = np.std(U_prime)
            j_idx = [n['id'] for n in all_nodes].index(node_spec['id'])
            BETA_HOTSPOT = 1.0  # 1σ above mean is considered hotspot
            GAMMA_HOTSPOT = 1.0  # Penalty weight (increased from 0.5 to strongly penalize hotspots)

            hotspot_threshold = mean_u + BETA_HOTSPOT * std_u
            hotspot_penalty = max(0.0, U_prime[j_idx] - hotspot_threshold) ** 2

            # Total stability cost = Gini coefficient + hotspot penalty
            return gini + GAMMA_HOTSPOT * hotspot_penalty

        except Exception as e:
            # Fallback to simple heuristic if Gini calculation fails
            status = self.get_node_current_status(node_spec['id'])
            current_load = status['used_flops'] / node_spec['compute_power'] if node_spec['compute_power'] > 0 else 1.0
            task_load_impact = self._get_execution_time(task_spec, node_spec) / self.avg_task_duration_fallback
            return current_load + max(0, current_load + task_load_impact - 0.8) * 10.0

    def _calculate_locality_cost(self, task_spec: Dict, node_spec: Dict) -> float:
        """Estimate expected communication penalty for placing the task on `node_spec`."""
        if not task_spec or not node_spec:
            return 0.0

        task_id = task_spec.get('id')
        candidate_node_id = node_spec.get('id')
        predecessors = self.task_reverse_deps.get(task_id, [])
        total_cost = 0.0
        total_weight = 0.0

        for pred_id in predecessors:
            pred_spec = self.task_map.get(pred_id, {})
            pred_state = self.get_task_state(pred_id, {}) if hasattr(self, 'get_task_state') else {}
            pred_node_id = pred_state.get('node_assigned', pred_state.get('node_id'))
            data_size_mb = pred_spec.get('data_size', 0)
            # FIX: data_size is already in MB, not bytes — removed erroneous /(1024*1024)
            weight = max(1.0, data_size_mb if data_size_mb else 0.0)

            if pred_node_id and pred_node_id != candidate_node_id:
                pred_node_spec = self._node_lookup_cache.get(pred_node_id)
                if pred_node_spec:
                    comm_time = self._calculate_comm_time(pred_spec, pred_node_spec, node_spec)
                else:
                    comm_time = self._AVG_LATENCY_MS / 1000.0
            else:
                comm_time = 0.0

            total_cost += comm_time * weight
            total_weight += weight

        reservations = self._get_reservations_for_task(task_id)
        if reservations:
            reserved_node_ids = {r.node_id for r in reservations}
            if candidate_node_id not in reserved_node_ids:
                avg_span = sum((r.end_time - r.start_time) for r in reservations) / len(reservations)
                total_cost += max(0.0, avg_span)
                total_weight += 1.0

        return total_cost / total_weight if total_weight > 0 else 0.0

    def _select_bandit_modifiers(self, dc_norm: float, effective_load: float) -> Dict[str, float]:
        if not self.bandit_enabled or not self.bandit_arms:
            self.bandit_active_arm = None
            self.bandit_last_context = None
            return {}
        total = sum(self.bandit_counts)
        explore = random.random() < self.bandit_epsilon
        if explore:
            idx = random.randrange(len(self.bandit_arms))
        else:
            denom = max(total, 1.0) + 1.0
            scores = []
            for i, value in enumerate(self.bandit_values):
                bonus = math.sqrt(2.0 * math.log(denom) / self.bandit_counts[i])
                scores.append(value + bonus)
            idx = max(range(len(scores)), key=lambda i: scores[i])
        self.bandit_active_arm = idx
        self.bandit_last_context = (dc_norm, effective_load)
        return dict(self.bandit_arms[idx])

    def _update_bandit(self, reward: float):
        if not self.bandit_enabled or self.bandit_active_arm is None or not self.bandit_arms:
            self.bandit_active_arm = None
            self.bandit_last_context = None
            return
        idx = self.bandit_active_arm
        self.bandit_counts[idx] += 1.0
        adjusted_reward = max(reward, 0.0)
        if self.bandit_last_context:
            dc_norm, effective_load = self.bandit_last_context
            adjusted_reward *= 1.0 + 0.15 * dc_norm
            adjusted_reward *= 1.0 + 0.1 * (1.0 - min(1.0, effective_load))
        step = 1.0 / self.bandit_counts[idx]
        self.bandit_values[idx] += step * (adjusted_reward - self.bandit_values[idx])
        decay = self.bandit_reward_decay
        if decay > 0.0:
            factor = 1.0 - decay * step
            for i in range(len(self.bandit_values)):
                if i == idx:
                    continue
                self.bandit_values[i] *= max(0.0, factor)
        self.bandit_active_arm = None
        self.bandit_last_context = None

    def _compute_bandit_reward(self, total_cost: float, dc_norm: float, effective_load: float) -> float:
        base = 1.0 / (1.0 + max(0.0, total_cost))
        base *= 1.0 + 0.2 * dc_norm
        base *= 1.0 + 0.1 * (1.0 - min(1.0, effective_load))
        return base

    def _compute_upward_ranks(self) -> Dict[str, float]:
        """Calculates upward rank (rank_u) for all tasks using per-task execution time (Defect G fix)."""
        ranks = {}

        def get_et(td):
            """Get per-task execution time ET = W_i / ReqC_i."""
            spec = self.task_map.get(td, {})
            if 'duration' in spec and spec['duration'] and spec['duration'] > 0:
                return spec['duration']
            w = spec.get('compute_demand', 1.0)
            reqc = spec.get('compute_rate', None)
            if reqc and reqc > 0:
                return w / reqc
            return self.avg_task_duration_fallback

        topo_order = self.topological_sort()
        for task_id in reversed(topo_order):
            task_et = get_et(task_id)
            max_successor_cost = 0
            for succ_id in self.task_deps.get(task_id, []):
                avg_comm_time = self._calculate_avg_comm_time_between_tasks(task_id, succ_id)
                successor_cost = avg_comm_time + ranks.get(succ_id, get_et(succ_id))
                max_successor_cost = max(max_successor_cost, successor_cost)
            ranks[task_id] = task_et + max_successor_cost
        self.max_upward_rank = max(ranks.values()) if ranks else 1.0
        return ranks

    def _get_feasible_nodes(self, task_spec: Dict) -> List[Dict]:
        """Filters nodes where task execution is possible based on requirements.
        Uses caching to avoid repeated feasibility checks for identical task requirements."""
        # Create cache key based on task requirements
        cache_key = (
            task_spec.get('compute_demand', 1.0),
            task_spec.get('memory_demand', 1.0),
            task_spec.get('model_required', 'default')
        )
        
        # Return cached result if available
        if cache_key in self._feasible_nodes_cache:
            return self._feasible_nodes_cache[cache_key]
        
        # Input validation
        if not task_spec:
            return []
            
        feasible_nodes = []
        for node_spec in self.nodes_spec:
            try:
                res_man = self.node_resources[node_spec['id']]
                exceeds, reason = res_man.exceeds_system_limits(
                    task_spec.get('compute_demand', 1.0),
                    task_spec.get('memory_demand', 1.0) * 1024,
                    task_spec.get('model_required', 'default')
                )
                if not exceeds:
                    feasible_nodes.append(node_spec)
                elif self.debug_cpls:
                    print(f"[FEASIBILITY] task {task_spec.get('id')} cannot run on node {node_spec.get('id')}: {reason}")
            except (KeyError, AttributeError) as e:
                # Skip problematic nodes but log warning
                if self.log_manager:
                    self.log_manager.log_print(f"Warning: Node {node_spec.get('id', 'unknown')} feasibility check failed: {e}")
                continue
        
        # Cache result for future use
        self._feasible_nodes_cache[cache_key] = feasible_nodes
        return feasible_nodes

    def _get_top_k_feasible_nodes(self, task_spec: Dict, k: int, mode: str = "eft") -> List[Dict]:
        """Gets feasible nodes and returns top-k candidates.
        
        Args:
            task_spec: Task specification
            k: Number of candidates to return
            mode: "eft" for EFT-first (OCAP default), "cpls" for CPLS-aware scoring
                  that considers load balance and communication to help CPLS
                  find globally better placements (not just locally optimal EFT).
        """
        candidates = self._get_feasible_nodes(task_spec)
        if not candidates:
            return []
        scored = []
        for node in candidates:
            eft, st, et = self._calculate_immediate_cost(task_spec, node)
            if eft == float('inf'):
                continue
            if mode == "cpls":
                # CPLS scoring: penalize heavily-loaded nodes to diversify
                # from OCAP's pure EFT-first selection. This makes CPLS
                # potentially choose a different node than OCAP, so that
                # its reservation actually constrains OCAP's later choices.
                load_penalty = 0.0
                try:
                    status = self.get_node_current_status(node['id'])
                    used = status.get('used_flops', 0) / max(node.get('compute_power', 1), 0.01)
                    load_penalty = max(0.0, used - 0.5) * eft  # penalize high-load nodes
                except Exception:
                    pass
                # Communication cost to highest-rank successor
                comm_bonus = 0.0
                succs = self.task_deps.get(task_spec.get('id', ''), [])
                if succs:
                    best_succ_rank = max(self.upward_ranks.get(s, 0) for s in succs)
                    succ_spec = max(succs, key=lambda s: self.upward_ranks.get(s, 0))
                    succ_spec_d = self.task_map.get(succ_spec, {})
                    succ_nodes = self._get_feasible_nodes(succ_spec_d)
                    if succ_nodes:
                        min_comm = min(
                            (self._calculate_comm_time(task_spec, node, sn) for sn in succ_nodes[:3]),
                            default=0
                        )
                        comm_bonus = min_comm * (best_succ_rank / max(self.max_upward_rank, 1.0))
                # Cloud-node preference: critical path tasks benefit from cloud's
                # high compute power and low inter-cloud latency for chain execution.
                # This creates genuine scheduling divergence: CPLS prefers cloud for
                # critical tasks, while OCAP (without this bias) may choose edge/end.
                # P1-2 FIX: Always apply cloud preference for CPLS, not gated by
                # cpls_soft_reservation — critical tasks always benefit from cloud's
                # high compute power regardless of reservation mode.
                cloud_bonus = 0.0
                node_type = node.get('type', '')
                if node_type == 'cloud':
                    cloud_bonus = -0.25 * eft  # prefer cloud for critical tasks
                elif node_type == 'edge':
                    cloud_bonus = 0.05 * eft  # slight penalty for edge
                cpls_score = eft + load_penalty + comm_bonus + cloud_bonus
                scored.append((cpls_score, -node.get('compute_power', 0), node))
            else:
                scored.append((eft, -node.get('compute_power', 0), node))
        if not scored:
            return []
        scored.sort(key=lambda x: (x[0], x[1]))
        return [n for _, __, n in scored[:k]]

    def _calculate_avg_network_transfer_time(self, data_size_mb: float) -> float:
        """Estimate average transfer delay over sampled node pairs for rank/path communication terms."""
        cache_key = round(float(data_size_mb), 6)
        if cache_key in self._avg_pair_comm_cache:
            return self._avg_pair_comm_cache[cache_key]

        fallback = (self._AVG_LATENCY_MS / 1000.0) + (data_size_mb / self._AVG_BANDWIDTH_MBPS)
        if not self.network_simulator or not self.use_network_tt:
            self._avg_pair_comm_cache[cache_key] = fallback
            return fallback

        node_ids = [n.get('id') for n in self.nodes_spec if isinstance(n, dict) and n.get('id')]
        if len(node_ids) <= 1:
            self._avg_pair_comm_cache[cache_key] = fallback
            return fallback

        pairs = [(src, dst) for src in node_ids for dst in node_ids if src != dst]
        if not pairs:
            self._avg_pair_comm_cache[cache_key] = fallback
            return fallback

        ratio = self.network_pair_sample_ratio
        if ratio <= 0.0:
            sampled_pairs = pairs[:1]
        elif ratio >= 1.0:
            sampled_pairs = pairs
        else:
            step = max(1, int(round(1.0 / ratio)))
            sampled_pairs = pairs[::step]

        delays: List[float] = []
        for src, dst in sampled_pairs:
            try:
                d = self.network_simulator.get_transfer_delay(src, dst, data_size_mb)
                if isinstance(d, (int, float)) and math.isfinite(d) and d >= 0.0:
                    delays.append(float(d))
            except Exception:
                continue

        avg_delay = (sum(delays) / len(delays)) if delays else fallback
        self._avg_pair_comm_cache[cache_key] = avg_delay
        return avg_delay

    def _get_execution_time(self, task_spec: Dict, node_spec: Dict) -> float:
        """Calculates execution time of a task on a specific node.
        Uses caching to avoid repeated calculations for same task-node pairs.
        """
        # Input validation
        if not task_spec or not node_spec:
            return float('inf')
            
        # Create cache key
        cache_key = (task_spec.get('id', ''), node_spec.get('id', ''))
        
        # Return cached result if available
        if cache_key in self._exec_time_cache:
            return self._exec_time_cache[cache_key]
        
        # Prefer runtime duration hint (matches NodeScheduler behavior)
        task_id = task_spec.get('id')
        st = self.get_task_state(task_id, {}) if task_id else {}
        dur_hint = st.get('duration_hint')
        if dur_hint and dur_hint > 0:
            exec_time = dur_hint
        else:
            exec_time = task_spec.get('duration', self.avg_task_duration_fallback)
            if not exec_time or exec_time <= 0:
                compute_demand = task_spec.get('compute_demand', 1.0)
                compute_power = node_spec.get('compute_power', 1.0)
                exec_time = max(0.001, compute_demand / max(1e-6, compute_power))
        
        # Cache result
        self._exec_time_cache[cache_key] = exec_time
        return exec_time

    def _calculate_data_ready_time(self, task_spec: Dict, node_id: str) -> float:
        """Calculates when all predecessor data will be available at the target node.
        Optimized with node lookup cache to avoid O(n) searches.
        """
        # Input validation
        if not task_spec or not node_id:
            return self.env.now if self.env else 0.0
            
        max_arrival_time = 0.0
        task_id = task_spec.get('id', '')
        
        for pred_id in self.task_reverse_deps.get(task_id, []):
            try:
                pred_state = self.get_task_state(pred_id, {})
                pred_finish_time = pred_state.get('completion_time') or pred_state.get('finish_time') or (self.env.now if self.env else 0.0)
                pred_node_id = pred_state.get('node_assigned', pred_state.get('node_id'))

                if pred_node_id == node_id or pred_node_id is None:
                    arrival_time = pred_finish_time
                else:
                    # Use optimized node lookup from cache
                    pred_node_spec = self._node_lookup_cache.get(pred_node_id)
                    target_node_spec = self._node_lookup_cache.get(node_id)
                    
                    if pred_node_spec and target_node_spec:
                        comm_time = self._calculate_comm_time(self.task_map.get(pred_id, {}), pred_node_spec, target_node_spec)
                        arrival_time = pred_finish_time + comm_time
                    else:
                        # Fallback: assume local execution if node lookup fails
                        arrival_time = pred_finish_time
                        
                max_arrival_time = max(max_arrival_time, arrival_time)
            except (KeyError, AttributeError) as e:
                # Skip problematic predecessors but continue processing
                if self.log_manager:
                    self.log_manager.log_print(f"Warning: Failed to process predecessor {pred_id}: {e}")
                continue
                
        return max_arrival_time

    def _calculate_comm_time(self, source_task_spec: Dict, source_node: Dict, dest_node: Dict) -> float:
        """Calculates communication time between two nodes for a given data size.
        Uses caching to avoid repeated network parameter lookups.
        """
        # Fast path for local communication or invalid inputs
        if (not source_node or not dest_node or 
            source_node.get('id') == dest_node.get('id')):
            return 0.0
        
        # Create cache key for node pair and data size
        data_size = source_task_spec.get('data_size', 0)
        cache_key = (source_node.get('id', ''), dest_node.get('id', ''), data_size)
        
        # Return cached result if available
        if cache_key in self._comm_time_cache:
            return self._comm_time_cache[cache_key]
        
        # FIX: data_size is already in MB, not bytes
        data_size_mb = data_size if data_size > 0 else 0.0
        if self.network_simulator and self.use_network_tt:
            comm_time = self.network_simulator.get_transfer_delay(source_node.get('id', ''), dest_node.get('id', ''), data_size_mb)
        else:
            latency_ms, bandwidth_mbps = self._estimate_network_params(source_node, dest_node)
            if bandwidth_mbps <= 0:
                comm_time = float('inf')
            else:
                comm_time = (latency_ms / 1000.0) + (data_size_mb / bandwidth_mbps)
        
        # Cache result
        self._comm_time_cache[cache_key] = comm_time
        return comm_time

    def _get_task_duration(self, task_id: str) -> float:
        st = self.get_task_state(task_id, {})
        dur = st.get('duration_hint')
        if dur and dur > 0:
            return dur
        spec = self.task_map.get(task_id, {})
        dur2 = spec.get('duration', self.avg_task_duration_fallback)
        return dur2 if dur2 and dur2 > 0 else self.avg_task_duration_fallback

    def _estimate_wait_time(self, res_man, task_spec: Dict) -> float:
        compute_demand = task_spec.get('compute_demand', 1.0)
        memory_mb = task_spec.get('memory_demand', 1.0) * 1024
        # Prefer NodeScheduler's queue-aware prediction if available
        try:
            ns = None
            # res_man carries node_id
            node_id = getattr(res_man, 'node_id', None)
            if node_id and hasattr(self, 'node_schedulers') and self.node_schedulers:
                ns = self.node_schedulers.get(node_id)
            if ns and hasattr(ns, 'predict_wait_time_for'):
                return max(0.0, ns.predict_wait_time_for(compute_demand, memory_mb, now=self.env.now))
        except Exception:
            pass
        # Fallback heuristic
        cpu_gap = max(0.0, compute_demand - max(0.0, res_man.total_flops - res_man.used_flops))
        mem_gap = max(0.0, memory_mb - max(0.0, res_man.total_memory - res_man.used_memory))
        cpu_ratio = cpu_gap / float(res_man.total_flops) if res_man.total_flops > 0 else 1.0
        mem_ratio = mem_gap / float(res_man.total_memory) if res_man.total_memory > 0 else 1.0
        occ = max(cpu_ratio, mem_ratio)
        base = self.avg_task_duration_fallback
        return max(0.0, occ * base)

    def _estimate_network_params(self, n1: Dict, n2: Dict) -> Tuple[float, float]:
        """Estimates latency (ms) and bandwidth (MB/s) between two nodes based on type.
        Optimized with constant lookup table for better performance.
        """
        # Input validation
        if not n1 or not n2:
            return self._NETWORK_PARAMS['default']
            
        t1, t2 = n1.get('type', 'unknown'), n2.get('type', 'unknown')
        
        # Check intra-type communication first (most common case)
        if t1 == t2:
            return self._NETWORK_PARAMS.get((t1, t1), self._NETWORK_PARAMS['default'])
        
        # Check inter-type communication (normalize order for consistent lookup)
        node_types = tuple(sorted([t1, t2]))
        if node_types in self._NETWORK_PARAMS:
            return self._NETWORK_PARAMS[node_types]
        
        # Default case
        return self._NETWORK_PARAMS['default']

    def _get_reservations_for_task(self, task_id: str) -> List[Reservation]:
        """Returns non-stale reservations created for the given task across nodes."""
        results: List[Reservation] = []
        if not task_id:
            return results
        now = self.env.now if self.env else 0.0
        try:
            for res_list in self.reservation_table.reservations.values():
                for r in res_list:
                    if r.task_id == task_id and r.end_time >= now:
                        results.append(r)
        except Exception:
            return []
        results.sort(key=lambda r: r.start_time)
        return results

    def _get_next_reservation_for_node(self, node_id: str, reference_time: float) -> Optional[Reservation]:
        """Fetch the nearest future reservation on the specified node."""
        if not node_id:
            return None
        return self.reservation_table.get_next_reservation(node_id, reference_time)

    def _calculate_system_load(self) -> float:
        """Enhanced system load calculation with bottleneck detection and history tracking."""
        if not self.node_resources:
            return 0.0
        
        total_flops_used = 0
        total_flops_capacity = 0
        total_memory_used = 0
        total_memory_capacity = 0
        node_loads = []
        
        for node_id, res_man in self.node_resources.items():
            total_flops_used += res_man.used_flops
            total_flops_capacity += res_man.total_flops
            total_memory_used += res_man.used_memory
            total_memory_capacity += res_man.total_memory
            
            # Track individual node loads for bottleneck detection
            if res_man.total_flops > 0 and res_man.total_memory > 0:
                node_flops_load = res_man.used_flops / res_man.total_flops
                node_memory_load = res_man.used_memory / res_man.total_memory
                node_load = max(node_flops_load, node_memory_load)
                node_loads.append((node_id, node_load))
                
                # Mark heavily loaded nodes as bottlenecks
                if node_load > 0.9:
                    self.bottleneck_nodes.add(node_id)
                elif node_load < 0.7:
                    self.bottleneck_nodes.discard(node_id)
        
        if total_flops_capacity == 0 or total_memory_capacity == 0:
            return 0.0
        
        flops_utilization = total_flops_used / total_flops_capacity
        memory_utilization = total_memory_used / total_memory_capacity
        system_load = max(flops_utilization, memory_utilization)
        
        # Update load history for trend analysis (keep last 10 measurements)
        self.system_load_history.append(system_load)
        if len(self.system_load_history) > 10:
            self.system_load_history.pop(0)
        
        # Adjust load calculation based on load distribution variance
        if len(node_loads) > 1:
            loads_only = [load for _, load in node_loads]
            load_variance = sum((x - system_load) ** 2 for x in loads_only) / len(loads_only)
            # High variance indicates uneven distribution, increase effective load
            if load_variance > 0.1:
                system_load *= (1.0 + min(0.2, load_variance))
        
        return system_load

    def _predict_workload_trend(self) -> float:
        """Predict future workload trend based on historical data."""
        if not self.workload_prediction_enabled or len(self.system_load_history) < 3:
            return 0.0
        
        # Simple linear trend prediction
        recent_loads = self.system_load_history[-self.prediction_window:]
        if len(recent_loads) < 2:
            return 0.0
        
        # Calculate trend slope
        trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        
        # Buffer trends for smoother prediction
        self.workload_trend_buffer.append(trend)
        if len(self.workload_trend_buffer) > 5:
            self.workload_trend_buffer.pop(0)
        
        # Return smoothed trend
        return sum(self.workload_trend_buffer) / len(self.workload_trend_buffer)
    
    def _predict_node_availability(self, node_id: str) -> float:
        """Predict when a node will become available based on current tasks."""
        if node_id not in self.node_schedulers:
            return 0.0
        
        node_scheduler = self.node_schedulers[node_id]
        current_time = self.env.now if self.env else 0.0
        
        # Get running tasks and their predicted finish times
        running_tasks = node_scheduler.running_tasks
        if not running_tasks:
            return current_time
        
        # Find the latest finish time
        latest_finish = current_time
        for task in running_tasks:
            if hasattr(task, 'predicted_finish_time'):
                latest_finish = max(latest_finish, task.predicted_finish_time)
            elif hasattr(task, 'start_time') and hasattr(task, 'duration'):
                predicted_finish = task.start_time + task.duration
                latest_finish = max(latest_finish, predicted_finish)
        
        return latest_finish

    def _calculate_avg_comm_time_between_tasks(self, task1_id: str, task2_id: str) -> float:
        """Calculates average communication time for upward rank calculation.
        Uses class constants for consistency and maintainability.
        """
        # Input validation
        if not task1_id or task1_id not in self.task_map:
            return 0.0
            
        task1_spec = self.task_map[task1_id]
        # FIX: data_size is already in MB, not bytes
        data_size_mb = task1_spec.get('data_size', 0)
        return self._calculate_avg_network_transfer_time(data_size_mb)
    
    def _calculate_avg_comm_to_successors(self, current_task_spec: Dict, successor_task_spec: Dict, current_node: Dict) -> float:
        """Calculates expected communication time from current node to potential locations of successor.
        Uses class constants and improved input validation.
        """
        # Input validation
        if not current_task_spec:
            return 0.0
            
        # FIX: data_size is already in MB, not bytes
        data_size_mb = current_task_spec.get('data_size', 0)
        return self._calculate_avg_network_transfer_time(data_size_mb)

    def _estimate_avg_comm_time(self, source_task_id: str, target_task_id: str) -> float:
        """
        Estimates average communication time between tasks, considering data size.
        Uses existing network parameter constants.
        """
        if not source_task_id or source_task_id not in self.task_map:
            return 0.0
            
        source_task_spec = self.task_map[source_task_id]
        # FIX: data_size is already in MB, not bytes
        data_size_mb = source_task_spec.get('data_size', 0)
        return self._calculate_avg_network_transfer_time(data_size_mb)

    def _identify_critical_path_segment(self, start_task_id: str, depth: int) -> List[str]:
        """
        Identifies a critical path segment starting from start_task_id.
        Relies on pre-computed upward ranks.
        Returns a list of task IDs.
        """
        if depth <= 0:
            return []

        # Debug logging
        if self.log_manager and (self.log_manager.verbose or self.debug_cpls) and depth > 0:
            self.log_manager.log_print(f"CPLS path segment starting from task {start_task_id}, depth={depth}")

        path = []
        current_task_id = start_task_id
        visited = {start_task_id}

        for _ in range(depth):
            path.append(current_task_id)

            successors = self.task_deps.get(current_task_id, [])
            if not successors: 
                break # Reached end of path

            successor_scores = []
            for succ_id in successors:
                if succ_id in visited:
                    continue
                # P1-2 FIX: 后继选择仅考虑静态 rank（succ_rank），移除 comm_time，对齐论文 Algorithm 2
                # BEFORE: score = comm_time + succ_rank
                succ_rank = self.upward_ranks.get(succ_id, 0)
                score = succ_rank
                successor_scores.append((score, succ_id))

            best_successor_id = None
            if successor_scores:
                successor_scores.sort(reverse=True)
                mode = (self.successor_mode or "max_rank").lower()
                if mode == "stochastic":
                    top_k = successor_scores[:max(1, min(self.successor_top_k, len(successor_scores)))]
                    weights = [max(1e-9, s) for s, _ in top_k]
                    candidates = [sid for _, sid in top_k]
                    best_successor_id = random.choices(candidates, weights=weights, k=1)[0]
                elif mode == "top_k":
                    top_k = successor_scores[:max(1, min(self.successor_top_k, len(successor_scores)))]
                    best_successor_id = top_k[0][1]
                else:
                    best_successor_id = successor_scores[0][1]

            if best_successor_id:
                visited.add(best_successor_id)
                current_task_id = best_successor_id
            else:
                break # No valid successor found

        # Debug logging
        if self.log_manager and (self.log_manager.verbose or self.debug_cpls) and depth > 0:
            self.log_manager.log_print(f"CPLS path segment result: {len(path)} tasks, path: {path}")

        return path

    def _greedy_fallback(self, task_spec: Dict) -> Optional[Dict]:
        """Greedy fallback when both CPLS and OCAP fail.
        Selects the node with highest compute power and lowest current load.
        This is part of DREAM's two-tier scheduling framework:
        OCAP (multi-objective optimization) → greedy fallback (best-effort placement).
        """
        try:
            best_node = None
            best_score = -1.0
            task_id = task_spec.get('id', 'unknown')

            compute_demand = task_spec.get('compute_demand', 1.0)
            memory_demand = task_spec.get('memory_demand', 1.0)
            model_required = task_spec.get('model_required', 'default')

            for node_spec in self.nodes_spec:
                try:
                    node_compute_power = node_spec.get('compute_power', 1.0)
                    node_memory = node_spec.get('memory', 1.0)

                    if node_compute_power <= 0 or node_memory <= 0:
                        continue

                    # Check basic resource compatibility
                    try:
                        res_man = self.node_resources.get(node_spec['id'])
                        if res_man:
                            exceeds, _ = res_man.exceeds_system_limits(
                                compute_demand,
                                memory_demand * 1024,
                                model_required
                            )
                            if exceeds:
                                continue
                    except Exception:
                        if compute_demand > node_compute_power or memory_demand > node_memory:
                            continue

                    # Get current load
                    try:
                        status = self.get_node_current_status(node_spec['id'])
                        current_cpu_load = status.get('used_flops', 0) / node_compute_power
                        current_mem_load = status.get('used_memory', 0) / (node_memory * 1024)
                        current_load = min(max(current_cpu_load, current_mem_load), 1.0)
                    except Exception:
                        current_load = 0.5

                    # Score: prefer high power, low load nodes
                    load_penalty = max(0, current_load - 0.8) * 5.0
                    # Balance factor: penalize nodes above cluster average
                    balance_bonus = 0.0
                    try:
                        all_loads = []
                        for ns in self.nodes_spec:
                            try:
                                ns_status = self.get_node_current_status(ns['id'])
                                ns_cpu = ns_status.get('used_flops', 0) / ns['compute_power'] if ns['compute_power'] > 0 else 0.0
                                all_loads.append(ns_cpu)
                            except Exception:
                                pass
                        if all_loads:
                            mean_load = sum(all_loads) / len(all_loads)
                            balance_bonus = (mean_load - current_load) * node_compute_power * 0.3
                    except Exception:
                        pass
                    score = node_compute_power * (1.0 - current_load) - load_penalty + balance_bonus

                    if score > best_score:
                        best_score = score
                        best_node = node_spec

                except Exception:
                    continue

            if not best_node and self.nodes_spec:
                best_node = self.nodes_spec[0]

            if self.log_manager and self.log_manager.verbose:
                self.log_manager.log_print(f"Greedy fallback for task {task_id}: selected {best_node['id'] if best_node else 'None'}")

            return best_node

        except Exception:
            return self.nodes_spec[0] if self.nodes_spec else None
