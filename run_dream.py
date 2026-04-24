#!/usr/bin/env python3
"""run_dream.py - Minimal DREAM scheduling simulation runner.

Runs the DREAM proactive scheduler (or HEFT baseline) on a cloud-edge-end
heterogeneous network with dynamic resource degradation events.

Usage:
    python run_dream.py                           # DREAM with showcase scenario
    python run_dream.py --strategy heft           # HEFT baseline comparison
    python run_dream.py --experiment scaling       # Parameter scaling scan
    python run_dream.py --strategy dream_proactive --verbose  # Verbose output
"""

from __future__ import annotations

import argparse
import json
import random
import simpy
import statistics
from typing import Dict, List, Tuple

from core.cluster_manager import ClusterManager
from core.log_manager import LogManager
from core.network_sim import NetworkSimulator
from generators.task_generator import generate_dynamic_task_dag, dynamic_task_dag_to_json
from generators.topology_generator import generate_resource_structure, create_network_topology, network_to_json
from project_paths import ensure_results_layout


# ---------------------------------------------------------------------------
# Resource degradation events (identical for all strategies)
# ---------------------------------------------------------------------------

def _schedule_resource_events(env: simpy.Environment, cluster_manager: ClusterManager, log_manager: LogManager):
    """Dynamic resource degradation events that stress-test scheduling adaptability.

    Events are identical for ALL strategies, ensuring fair experimental environment.
    Uses dynamic node selection based on type to work with any topology.
    """
    # Classify nodes by type for targeted degradation
    cloud_nodes = [n['id'] for n in cluster_manager.nodes_spec if n.get('type') == 'cloud']
    edge_nodes = [n['id'] for n in cluster_manager.nodes_spec if n.get('type') == 'edge']
    end_nodes = [n['id'] for n in cluster_manager.nodes_spec if n.get('type') == 'end']

    def _capacity_event(delay, node_id, flops_factor, memory_factor, duration, label):
        yield env.timeout(delay)
        res_man = cluster_manager.node_resources.get(node_id)
        if not res_man:
            return
        if log_manager and log_manager.verbose:
            log_manager.log_print(f"[{env.now:.2f}] Event '{label}': applying capacity factor "
                                  f"({flops_factor:.2f} FLOPS, {memory_factor:.2f} MEM) on {node_id}")
        res_man.apply_capacity_adjustment(flops_factor=flops_factor, memory_factor=memory_factor)
        yield env.timeout(duration)
        res_man.restore_capacity()
        if log_manager and log_manager.verbose:
            log_manager.log_print(f"[{env.now:.2f}] Event '{label}': restored capacity on {node_id}")

    def _rolling_throttle(start_time, interval, reductions):
        yield env.timeout(start_time)
        while True:
            for node_id, flops_factor, mem_factor in reductions:
                res_man = cluster_manager.node_resources.get(node_id)
                if not res_man:
                    continue
                jitter = random.uniform(0.8, 1.2)
                res_man.apply_capacity_adjustment(flops_factor=flops_factor * jitter, memory_factor=mem_factor)
            yield env.timeout(interval)
            for node_id, _, _ in reductions:
                res_man = cluster_manager.node_resources.get(node_id)
                if res_man:
                    res_man.restore_capacity()
            yield env.timeout(interval)

    # t=35: first edge node hotspot (45% capacity loss for 50 time units)
    if edge_nodes:
        env.process(_capacity_event(35.0, edge_nodes[0], 0.55, 0.6, 50.0, "edge_hotspot"))
    # t=90: first cloud node maintenance (35% capacity loss for 40 time units)
    if cloud_nodes:
        env.process(_capacity_event(90.0, cloud_nodes[0], 0.65, 0.8, 40.0, "cloud_maintenance"))
    # t=140: last edge node fault (60% capacity loss for 60 time units)
    if len(edge_nodes) > 1:
        env.process(_capacity_event(140.0, edge_nodes[-1], 0.4, 0.5, 60.0, "edge_fault"))
    # t=60+: rolling throttle on remaining edge nodes
    throttle_targets = [(nid, 0.8, 0.85) for nid in edge_nodes[1:-1]] if len(edge_nodes) > 2 else []
    if throttle_targets:
        env.process(_rolling_throttle(60.0, 25.0, throttle_targets))


# ---------------------------------------------------------------------------
# Metrics collection (identical for all strategies)
# ---------------------------------------------------------------------------

def _collect_metrics(cluster_manager: ClusterManager, sim_duration: float) -> Dict:
    """Collect common metrics + strategy-specific stats if available."""
    completed_tasks = []
    failed_tasks = []
    cascaded_blocked = 0
    for task_id, state in cluster_manager.task_states.items():
        status = state.get('status', 'unknown')
        if status == 'completed':
            completed_tasks.append(state)
        elif status == 'failed':
            failed_tasks.append(state)
        elif status == 'pending':
            # Check if this task is cascade-blocked (any predecessor failed)
            for pred_id in cluster_manager.task_reverse_deps.get(task_id, []):
                if cluster_manager.task_states.get(pred_id, {}).get('status') == 'failed':
                    cascaded_blocked += 1
                    break

    makespan = 0.0
    if completed_tasks:
        makespan = max(t.get('completion_time', 0) for t in completed_tasks if t.get('completion_time'))

    completion_rate = len(completed_tasks) / max(len(cluster_manager.task_states), 1)

    # Load balance (Gini coefficient of node utilization)
    node_utils = []
    for node_id, res_man in cluster_manager.node_resources.items():
        if res_man.total_flops > 0:
            node_utils.append(res_man.used_flops / res_man.total_flops)
    balance = 1.0 - _gini(node_utils) if node_utils else 0.0

    throughput = len(completed_tasks) / max(makespan, 1.0) * 1000  # tasks per 1000 time units

    metrics = {
        "makespan": round(makespan, 2),
        "completion_rate": round(completion_rate, 4),
        "completed_tasks": len(completed_tasks),
        "failed_tasks": len(failed_tasks),
        "cascaded_blocked": cascaded_blocked,
        "total_tasks": len(cluster_manager.task_states),
        "balance": round(balance, 4),
        "throughput": round(throughput, 4),
    }

    # Strategy-specific stats (DREAM only)
    if hasattr(cluster_manager, 'scheduler') and hasattr(cluster_manager.scheduler, 'get_scheduling_stats'):
        metrics["dream_stats"] = cluster_manager.scheduler.get_scheduling_stats()

    return metrics


def _gini(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    mean_v = sum(values) / n
    if mean_v == 0:
        return 0.0
    sum_abs = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))
    return sum_abs / (2 * n * n * mean_v)


# ---------------------------------------------------------------------------
# Single simulation run (shared by all strategies)
# ---------------------------------------------------------------------------

def _find_capable_nodes(task_spec: Dict, node_specs: List[Dict]) -> List[str]:
    """Find nodes capable of running a task based on model and resource requirements."""
    capable = []
    if not isinstance(task_spec, dict):
        return capable
    for node_spec in node_specs:
        if not isinstance(node_spec, dict):
            continue
        if (task_spec.get("model_required") in node_spec.get("models", [])
                and float(node_spec.get("compute_power", 0)) >= float(task_spec.get("compute_demand", 0))
                and float(node_spec.get("memory", 0)) >= float(task_spec.get("memory_demand", 0))):
            capable.append(node_spec["id"])
    return capable


def _delayed_start_task(env: simpy.Environment, cluster_manager: ClusterManager, task_id: str, delay: float):
    """Wait for delay, then schedule the task."""
    yield env.timeout(delay)
    cluster_manager.schedule_task(task_id)


def run_simulation(strategy: str = 'dream_proactive', duration: float = 500.0,
                   verbose: bool = False, seed: int = 42) -> Dict:
    """Run a single simulation with the specified strategy.

    Both DREAM and HEFT use the exact same:
    - Network topology
    - Task DAG
    - Resource degradation events
    - Metrics collection
    """
    ensure_results_layout()
    random.seed(seed)

    env = simpy.Environment()
    log_manager = LogManager(clean_logs=False, verbose=verbose)

    # Network topology: 10 nodes (C2/E3/End5) — same for all strategies
    resource_structure = generate_resource_structure(2, 3, 5)
    network_config = network_to_json(create_network_topology(resource_structure))
    network_simulator = NetworkSimulator(env, network_config, log_manager)

    # Task DAG: 50 dynamically generated tasks — same for all strategies
    dag = generate_dynamic_task_dag(50, max_unfold_iterations=3)
    dag_json = dynamic_task_dag_to_json(dag)

    # ClusterManager (unified init path handles all strategies)
    scheduler_config = {
        'task_timeout_multiplier': 5.0,
    }
    if strategy == 'dream_proactive':
        scheduler_config.update({
            'cpls_branching': 3,
            'criticality_alpha': 1.8,
            'lambda_penalty': 5.0,
        })

    cluster_manager = ClusterManager(
        env=env,
        network=network_config,
        task_dag=dag_json,
        log_manager=log_manager,
        network_simulator=network_simulator,
        scheduler_strategy=strategy,
        scheduler_config=scheduler_config,
        w_immediate=0.44,   # Paper-aligned weights
        w_ripple=0.27,
        w_stability=0.17,
        w_opportunity=0.12,
    )

    cluster_manager.initialize_simulation()

    # Resource degradation events (same for all strategies)
    _schedule_resource_events(env, cluster_manager, log_manager)

    # Launch source tasks (tasks with no predecessors)
    source_task_ids = [tid for tid in cluster_manager.task_map
                       if not cluster_manager.task_map[tid].get('predecessors')]

    for task_id in source_task_ids:
        task_spec = cluster_manager.task_map.get(task_id, {})
        capable_nodes = _find_capable_nodes(task_spec, network_config.get('nodes', []))
        if capable_nodes:
            initial_node_id = random.choice(capable_nodes)
            cluster_manager.set_initial_task_node(task_id, initial_node_id)
        start_delay = random.uniform(0, 0.1)
        env.process(_delayed_start_task(env, cluster_manager, task_id, start_delay))

    # Monitoring process
    def _monitor():
        while True:
            yield env.timeout(50.0)
            if verbose:
                log_manager.log_print(f"[Monitor t={env.now:.0f}] "
                    f"completed={sum(1 for s in cluster_manager.task_states.values() if s.get('status')=='completed')}, "
                    f"running={sum(1 for s in cluster_manager.task_states.values() if s.get('status')=='running')}, "
                    f"ready={sum(1 for s in cluster_manager.task_states.values() if s.get('status')=='ready')}")
    env.process(_monitor())

    # Run simulation
    print(f"\n{'='*60}")
    print(f"Running simulation: strategy={strategy}, duration={duration}")
    print(f"{'='*60}")
    env.run(until=duration)

    # Collect metrics
    metrics = _collect_metrics(cluster_manager, duration)
    _print_results(strategy, metrics)

    return metrics


def _print_results(strategy: str, metrics: Dict):
    print(f"\n{'='*60}")
    print(f"  Results: {strategy}")
    print(f"{'='*60}")
    print(f"  Makespan:          {metrics['makespan']:.2f}")
    print(f"  Completion Rate:   {metrics['completion_rate']:.2%}")
    print(f"  Completed/Total:   {metrics['completed_tasks']}/{metrics['total_tasks']}")
    print(f"  Failed Tasks:      {metrics['failed_tasks']}")
    if metrics.get('cascaded_blocked', 0) > 0:
        print(f"  Cascade-Blocked:  {metrics['cascaded_blocked']}")
    print(f"  Load Balance:      {metrics['balance']:.4f}")
    print(f"  Throughput:        {metrics['throughput']:.4f} tasks/1000tu")
    if "dream_stats" in metrics:
        ds = metrics["dream_stats"]
        print(f"  --- DREAM Stats ---")
        print(f"  CPLS Calls:       {ds.get('cpls_calls', 0)}")
        print(f"  OCAP Calls:       {ds.get('ocap_calls', 0)}")
        print(f"  Critical Ratio:   {ds.get('critical_ratio', 0):.4f}")
        print(f"  CPLS Decision Time (mean): {ds.get('cpls_decision_time_mean_ms', 0):.2f} ms")
        print(f"  Reservation Overlaps: {ds.get('reservation_overlap_count', 0)}")
        print(f"  Active Reservations:  {ds.get('active_reservations', 0)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Experiment 2: Parameter Scaling Scan
# ---------------------------------------------------------------------------

def run_scaling_experiment():
    """Run DREAM vs HEFT across varying node and task counts."""
    print(f"\n{'#'*70}")
    print(f"  Parameter Scaling Experiment: DREAM vs HEFT")
    print(f"{'#'*70}\n")

    # Node scaling: vary node count, fixed 100 tasks
    print("--- Node Scaling (fixed 100 tasks) ---")
    print(f"{'Nodes':>6} | {'Strategy':>16} | {'Makespan':>10} | {'CompRate':>9} | {'Balance':>9}")
    print("-" * 65)

    for num_nodes in range(5, 40, 5):
        for strategy in ['dream_proactive', 'heft']:
            metrics = _run_scaling_single(strategy, num_nodes=num_nodes, num_tasks=100)
            print(f"{num_nodes:>6} | {strategy:>16} | {metrics['makespan']:>10.2f} | "
                  f"{metrics['completion_rate']:>8.2%} | {metrics['balance']:>9.4f}")
        print()

    # Task scaling: vary task count, fixed 10 nodes (C2/E3/End5)
    print("\n--- Task Scaling (fixed 10 nodes) ---")
    print(f"{'Tasks':>6} | {'Strategy':>16} | {'Makespan':>10} | {'CompRate':>9} | {'Balance':>9}")
    print("-" * 65)

    for num_tasks in range(20, 220, 20):
        for strategy in ['dream_proactive', 'heft']:
            metrics = _run_scaling_single(strategy, num_nodes=10, num_tasks=num_tasks)
            print(f"{num_tasks:>6} | {strategy:>16} | {metrics['makespan']:>10.2f} | "
                  f"{metrics['completion_rate']:>8.2%} | {metrics['balance']:>9.4f}")
        print()


def _run_scaling_single(strategy: str, num_nodes: int, num_tasks: int,
                         duration: float = 1500.0, seed: int = 42) -> Dict:
    """Run a single scaling experiment point with dynamically generated topology and DAG."""
    ensure_results_layout()
    random.seed(seed)

    env = simpy.Environment()
    log_manager = LogManager(clean_logs=False, verbose=False)

    # Generate network topology with specified node count (2:4:4 ratio for cloud:edge:end)
    num_cloud = max(2, num_nodes // 5)
    num_edge = max(2, num_nodes * 2 // 5)
    num_end = max(2, num_nodes - num_cloud - num_edge)
    resource_structure = generate_resource_structure(num_cloud, num_edge, num_end)
    network_config = network_to_json(create_network_topology(resource_structure))
    network_simulator = NetworkSimulator(env, network_config, log_manager)

    # Generate task DAG with specified task count
    dag = generate_dynamic_task_dag(num_tasks, max_unfold_iterations=4)
    dag_json = dynamic_task_dag_to_json(dag)

    scheduler_config = {
        'task_timeout_multiplier': 5.0,
    }
    if strategy == 'dream_proactive':
        scheduler_config.update({
            'cpls_branching': 3,
            'criticality_alpha': 1.8,
            'lambda_penalty': 5.0,
        })

    cluster_manager = ClusterManager(
        env=env,
        network=network_config,
        task_dag=dag_json,
        log_manager=log_manager,
        network_simulator=network_simulator,
        scheduler_strategy=strategy,
        scheduler_config=scheduler_config,
        w_immediate=0.44,
        w_ripple=0.27,
        w_stability=0.17,
        w_opportunity=0.12,
    )

    cluster_manager.initialize_simulation()

    # Apply resource degradation events (same for all strategies)
    _schedule_resource_events(env, cluster_manager, log_manager)

    source_tasks = [tid for tid in cluster_manager.task_map
                    if not cluster_manager.task_map[tid].get('predecessors')]
    for task_id in source_tasks:
        task_spec = cluster_manager.task_map.get(task_id, {})
        capable = _find_capable_nodes(task_spec, network_config.get('nodes', []))
        if capable:
            cluster_manager.set_initial_task_node(task_id, random.choice(capable))
        env.process(_delayed_start_task(env, cluster_manager, task_id, random.uniform(0, 0.1)))

    env.run(until=duration)
    return _collect_metrics(cluster_manager, duration)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DREAM Scheduling Simulation")
    parser.add_argument('--strategy', type=str, default='dream_proactive',
                        choices=['dream_proactive', 'heft'],
                        help='Scheduling strategy (default: dream_proactive)')
    parser.add_argument('--duration', type=float, default=500.0,
                        help='Simulation duration in time units (default: 500)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--experiment', type=str, default='showcase',
                        choices=['showcase', 'scaling'],
                        help='Experiment type (default: showcase)')
    args = parser.parse_args()

    if args.experiment == 'scaling':
        run_scaling_experiment()
    else:
        run_simulation(
            strategy=args.strategy,
            duration=args.duration,
            verbose=args.verbose,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
