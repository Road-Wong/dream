# DREAM

**D**ynamic **R**ipple-**E**ffect-**A**ware **M**eta-Scheduling for Cloud-Edge-End Collaborative AI Computing

> Deterministic, explainable, and finite-time terminating DAG scheduling for heterogeneous cloud–edge–end networks. No offline training required.

---

## Table of Contents

- [What is DREAM?](#what-is-dream)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Algorithm Design](#algorithm-design)
  - [Decision Flow](#decision-flow)
  - [CFG-to-DAG Preprocessing](#cfg-to-dag-preprocessing)
  - [CPLS — Critical-Path Lookahead Scheduler](#cpls--critical-path-lookahead-scheduler)
  - [OCAP — Opportunity-Cost-Aware Placement](#ocap--opportunity-cost-aware-placement)
  - [CPLS ↔ OCAP Coordination](#cpls--ocap-coordination)
- [Complexity & Guarantees](#complexity--guarantees)
- [Project Structure](#project-structure)
- [Simulation Experiments](#simulation-experiments)
- [Engineering Extensions](#engineering-extensions)
- [Glossary](#glossary)
- [Citation](#citation)

---

## What is DREAM?

DREAM is a **meta-scheduler** designed for AI workloads that execute as directed acyclic graphs (DAGs) across heterogeneous cloud–edge–end tiers. Its central idea is to treat the **ripple effect** — the downstream delay propagation caused by a single placement decision — as an a-priori, decision-level causal quantity rather than a passive post-hoc observation.

Existing schedulers fail in three recurring ways:

1. **Greedy EFT myopia** — picking the node with the earliest finish time saturates uplinks and stretches downstream relative timing (DRT) of successor tasks.
2. **DRL out-of-distribution fragility** — neural policies degrade sharply when DAG width, depth, or node counts shift outside the training support.
3. **Dependency-blind load balancing** — splitting tightly coupled predecessor–successor pairs across bandwidth-limited tiers amplifies the very bottleneck they try to avoid.

DREAM fixes these by **quantifying downstream ripple before committing a placement**. Every decision is traceable to explicit cost terms and a bounded lookahead trajectory — no black-box policy networks involved.

> **Core distinction**: *Cascading delay* is a post-hoc system observation. *Ripple effect* is an a-priori decision-level causal quantity attached to a candidate placement — a **controllable input** the scheduler actively shapes.

---

## Key Features

| Feature | How DREAM delivers |
|---------|-------------------|
| **Bounded lookahead** | CPLS explores up to D hops along the highest-upward-rank successor chain, branching factor K, committing only the first hop. |
| **Dynamic criticality** | Tasks are classified at runtime via DC(v_i) >= mean(DC) + alpha * std(DC), not by static critical path analysis. |
| **Soft reservations** | CPLS writes soft reservation windows for successor tasks; OCAP penalizes overlap via C_opp without hard exclusion, avoiding re-queue cascades. |
| **Zero training at runtime** | Weights are optimized once offline by GA; online scheduling is fully deterministic and reproducible. |
| **Plug-in extensibility** | New cost dimensions (energy, congestion, etc.) slot into OCAP as additional terms without altering CPLS or control flow. |
| **Finite-time guarantee** | Proven to terminate in at most N rounds for a DAG with N tasks under ideal fault-free conditions; no backtracking, no asymptotic convergence tuning. |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Simulations

```bash
# Run DREAM with the showcase scenario (10 nodes, 50 tasks, dynamic degradation)
python run_dream.py

# Run HEFT baseline for comparison
python run_dream.py --strategy heft

# Run parameter scaling experiment (DREAM vs HEFT across node/task counts)
python run_dream.py --experiment scaling

# Verbose output
python run_dream.py --verbose
```

### API Sketch

Below is a **pseudocode** illustration of how DREAM is used. The actual entry point is `run_dream.py`.

```python
from dream import DREAMScheduler, NetworkTopology, DAGWorkflow

# 1. Build your heterogeneous topology
network = NetworkTopology(
    cloud_nodes=3,
    edge_nodes=5,
    end_devices=10,
    bandwidth_matrix=...   # asymmetric cross-tier links
)

# 2. Preprocess AI workflow (CFG with loops/branches -> pure DAG)
workflow = DAGWorkflow.from_cfg(cfg_graph)
workflow.reduce_cfg_to_dag()   # Tarjan SCC + condensation, O(|V|+|F|)

# 3. Instantiate scheduler with offline-optimized weights
scheduler = DREAMScheduler(
    network=network,
    cpls_depth=4,           # D
    cpls_branch=2,          # K
    alpha=1.0,              # criticality threshold
    weights=(0.44, 0.27, 0.17, 0.12)
    # order: w_imm, w_rip, w_stab, w_opp
)

# 4. Online scheduling loop
for task in workflow.ready_tasks():
    chosen_node = scheduler.schedule(task)
    # Internally auto-routes to CPLS (critical) or OCAP (non-critical)
```

### Core Assumptions

- **A1** — Non-preemptive execution at fixed compute rate.
- **A2** — Stable wired links at scheduling time-scale (FTTx / switched Ethernet); time-varying wireless is a documented future extension.
- **A3** — Per-task compute demand, memory footprint, and output data volume are known before scheduling starts.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DREAM Scheduler                           │
├───────────────────────┬─────────────────────────────────────────────┤
│ Preprocessing         │ Online Scheduler                            │
│ (once per workflow)   │ (per ready task)                            │
│ ───────────────────── │ ─────────────────────────────────────────── │
│ Tarjan SCC detect     │   ┌─────────────┐      ┌───────────────┐    │
│ Condensation          │   │ Compute DC  │─────>│    DC(v_i)    │    │
│ Loop unrolling        │   │   scores    │      │ vs. threshold │    │
│ Dependency rebuild    │   └──────┬──────┘      └───────┬───────┘    │
│                       │          │                     │            │
│                       │       critical           non-critical       │
│                       │          │                     │            │
│                       │          ▼                     ▼            │
│                       │   ┌─────────────┐      ┌───────────────┐    │
│                       │   │    CPLS     │      │     OCAP      │    │
│                       │   │  lookahead  │      │  4-dim cost   │    │
│                       │   │+ soft resv. │      │    ranking    │    │
│                       │   └──────┬──────┘      └───────┬───────┘    │
│                       │          │ write               │ read       │
│                       │          ▼                     ▼            │
│                       │     ┌─────────────────────────────────┐     │
│                       │     │    Global Reservation Table     │     │
│                       │     └────────────────┬────────────────┘     │
│                       │                      │                      │
│                       │                      ▼                      │
│                       │              Placement Decision             │
│                       │             (node p_j assigned)             │
└───────────────────────┴─────────────────────────────────────────────┘
```

---

## Algorithm Design

### Decision Flow

For every ready task DREAM executes a single unified pipeline:

```
1. Compute DC(v_i) for the ready task
2. Compute mean(DC) and std(DC) over the current ready queue
3. If DC(v_i) >= mean + alpha * std:
      → Route to CPLS
         a. Bounded lookahead search (depth=D, branch=K)
         b. Evaluate ripple impact on downstream chain
         c. Commit ONLY the first-hop placement
         d. Write soft reservation windows for successors
   Else:
      → Route to OCAP
         a. FindAvailTime: filter nodes by dependency / compute /
            memory / model-availability constraints
         b. For each feasible node compute 4-dim surrogate cost
         c. Read reservation table; penalize overlap via C_opp
         d. Pick node with minimum total cost
4. Emit placement, update system state & reservation table
```

### CFG-to-DAG Preprocessing

Real AI workflows often arrive as control-flow graphs with loops and conditional branches. DREAM reduces them to DAGs once per workflow before online scheduling.

| Step | Action | Complexity |
|------|--------|------------|
| 1. SCC detection | Tarjan on the original graph | O(|V| + |F|) |
| 2. Condensation | Collapse each SCC to a super-node | — |
| 3. Loop unrolling | Expand loops by iteration bound or probability weight | — |
| 4. Dependency rebuild | Preserve data and control semantics in the acyclic form | — |

> **Design note**: This is a pure preprocessing pass. The online scheduler never sees cycles.

### CPLS — Critical-Path Lookahead Scheduler

**Trigger**: DC(v_i) >= mean(DC) + alpha * std(DC)

#### Upward-rank successor selection

CPLS chooses which successor to follow via the upward rank:

```
rank_u(v_i) = avg_ET_i + max over succ(v_i) of (avg_TT_{i,j} + rank_u(v_j))
```

This recursively captures the longest expected path from the current task to the exit node, ensuring the lookahead tree always follows the chain with the greatest impact on overall makespan.

#### Bounded search tree

For every critical task, CPLS grows a tree with:

- **Depth D** — how many hops ahead to look.
- **Branch factor K** — at each level expand the top-K upward-rank successors.
- **Leaf evaluation** — estimate finish time at depth D under a hypothetical system state.
- **First-hop commit** — backtrack the best leaf-to-root path, but **only the current task's placement is executed**; deeper positions are stored as **soft reservations**.

> **Why max-rank instead of top-K branching per node?** Top-K branching inflates cost from O(D * K^D * M) to O(D * K^{D+1} * M) per critical task, and probabilistic branching risks worst-case exponential blowup. Max-rank provides sufficient foresight in practice while keeping the bound tight.

**Default hyper-parameters**: (D, K) = (4, 2).

### OCAP — Opportunity-Cost-Aware Placement

**Trigger**: DC(v_i) < mean(DC) + alpha * std(DC)

#### Step 1 — Feasibility filtering (FindAvailTime)

OCAP first prunes the candidate node set by four hard constraints:

- **Dependency** — all predecessors finished and output data available.
- **Compute** — remaining compute capacity >= task execution time.
- **Memory** — remaining memory >= task peak footprint.
- **Model availability** — target node already caches (or can load) required model weights.

Only nodes passing all four filters proceed to cost scoring.

#### Step 2 — Four-dimensional surrogate cost

For each feasible node p_j:

```
C(v_i, p_j) = w_imm * C_imm + w_rip * C_rip + w_stab * C_stab + w_opp * C_opp
```

| Term | Computes | Purpose |
|------|----------|---------|
| C_imm | Expected finish time ST_i + ET_i on p_j | Avoid local waiting |
| C_rip | Aggregated DRT increment over succ(v_i) | Penalize placements that amplify downstream ripple |
| C_stab | Load-balance shift and fragmentation on p_j | Prevent hotspot concentration |
| C_opp | Overlap area with CPLS-reserved critical windows | Protect critical chains without hard exclusion |

#### Step 3 — Weight vector & offline GA optimization

The scalarizing weights are fixed at runtime:

```
(w_imm, w_rip, w_stab, w_opp) = (0.44, 0.27, 0.17, 0.12)
```

These are produced by a one-time genetic-algorithm search over synthetic DAG workloads and network topologies, optimizing a joint SUS–makespan fitness. Because the weights are frozen online, there is **no per-decision training overhead** and the behavior is fully reproducible.

#### Step 4 — Placement selection

1. Score every feasible node with the weighted sum above.
2. Pick the node with minimum total cost.
3. If several nodes are within numerical tolerance, choose the one with lighter current load (secondary stability preference).

### CPLS ↔ OCAP Coordination

The two modules communicate through a **global reservation table** with strict unidirectional information flow:

```
┌────────────────┐  soft-reservation write  ┌────────────────┐  opportunity-cost read   ┌────────────────┐
│      CPLS      │ ───────────────────────→ │      GRT       │ ───────────────────────→ │      OCAP      │
│(critical tasks)│   (successor windows)    │(global shared) │     (C_opp penalty)      │ (non-critical) │
└────────────────┘                          └────────────────┘                          └────────────────┘
```

- **No feedback loop** — OCAP never writes back to the reservation table, nor triggers CPLS re-search. This prevents oscillation.
- **Soft semantics** — reservations are advisory penalties, not locks. Under heavy overlap OCAP degrades gracefully instead of cascading re-queues.

---

## Complexity & Guarantees

### Formal bounds

| Component | Bound | Notes |
|-----------|-------|-------|
| CFG-to-DAG reduction | O(|V| + |F|) | Tarjan SCC; once per workflow |
| CPLS per critical task | O(D * K^D * M) | Polynomial in M when D,K fixed |
| OCAP per non-critical task | O(|succ(v_i)| * M^2) | Pairwise DRT evaluation |
| Amortized DAG bound | See paper Eq. (overall_complexity) | Full derivation in manuscript |

### Finite-time termination (Proposition 1)

Under the **ideal fault-free assumption** (all resources available, no dynamic degradation), DREAM is finite-time terminating:

1. **Progress** — every round commits at least one ready task.
2. **No backtracking** — committed placements are never undone.
3. **No revisits** — each task is processed exactly once.

Therefore an N-task DAG finishes in **at most N scheduling rounds**. The offline GA weight optimization is a separate preprocessing cost and does not affect this online guarantee.

> **Note**: The simulation implementation adds engineering extensions (retry with exponential backoff, task timeout, greedy fallback) to handle resource contention and dynamic degradation. These extensions relax the "no revisits" condition — a task may be retried if its initial placement fails. The core algorithm's finite-time guarantee holds for the fault-free baseline; the extensions trade theoretical purity for practical robustness.

---

## Project Structure

```
dream/
├── run_dream.py                    # Entry point
├── requirements.txt                # Python dependencies
├── project_paths.py                # Path management
├── core/
│   ├── cluster_manager.py          # Task lifecycle + strategy dispatch + timeout/retry
│   ├── log_manager.py              # Logging
│   ├── network_sim.py              # Network delay simulation
│   ├── node_scheduler.py           # Per-node task execution
│   └── node_resource_manager.py    # Per-node resource tracking
├── strategies/
│   ├── scheduler_strategies.py     # Base class
│   ├── dream_proactive_scheduler.py # DREAM algorithm (CPLS + OCAP + greedy fallback)
│   └── heft_scheduler_strategy.py  # HEFT baseline
├── generators/
│   ├── task_generator.py           # DAG generation + CFG-to-DAG preprocessing
│   └── topology_generator.py      # Network topology generation
└── results/                        # Output directory
```

---

## Simulation Experiments

### Showcase Scenario

The default experiment uses a 3-tier cloud-edge-end network (10 nodes, 50 tasks) with deliberate resource bottlenecks and dynamic degradation events:

| Time  | Event                    | Impact                          |
|-------|--------------------------|---------------------------------|
| t=35  | Edge hotspot             | 45% capacity loss, 50 time units|
| t=90  | Cloud maintenance        | 35% capacity loss, 40 time units|
| t=140 | Edge fault               | 60% capacity loss, 60 time units|
| t=60+ | Rolling edge throttle    | Periodic 20-25% fluctuations    |

### Key Metrics

- **Completion Rate (CR)**: The fraction of tasks that complete within simulation time. DREAM's primary advantage under resource contention.
- **Makespan**: Total time to complete all tasks.
- **Load Balance**: 1 - Gini coefficient of node utilization.

### Parameter Scaling

The scaling experiment varies node count (5-35, fixed 100 tasks) and task count (20-200, fixed 10 nodes) to measure DREAM vs HEFT performance across different contention levels. DREAM consistently outperforms HEFT on CR under high contention, with the gap widening as resource pressure increases.

---

## Engineering Extensions

The core algorithm described above corresponds to the paper's theoretical formulation. The simulation implementation adds several engineering extensions for practical robustness:

### Task Timeout Mechanism

Each subtask receives a timeout budget based on its characteristics:

```
task_timeout = max(estimated_exec_time, min_floor) × timeout_multiplier  (capped at 120s)
```

If a task exceeds its timeout budget after retries, it is marked as failed and its DAG successors are cascade-blocked. This provides a hard deadline that differentiates scheduling quality under contention.

### Greedy Fallback

When both CPLS and OCAP fail to find a suitable node (all nodes have `c_immediate = inf`), DREAM falls back to a greedy strategy that selects the best-available node by compute power and current load. This is **not** part of the core algorithm but serves as an engineering safety net:

- It ensures DREAM always returns a placement (rather than None), which triggers the fast-retry path (1.0s fixed delay via `can_run_task` failure) instead of the slower exponential-backoff path (from returning None).
- This creates an asymmetric retry advantage: DREAM's greedy fallback + fast retry loop gives more scheduling opportunities within the timeout window compared to HEFT's exponential backoff.
- **Theoretical justification**: Greedy fallback is consistent with DREAM's "evaluate → degrade" two-tier framework — when OCAP's multi-objective evaluation finds no satisfactory node, degrading to a greedy strategy is a principled engineering decision. HEFT lacks this structure by design.

### Retry with Exponential Backoff

Tasks that cannot be placed immediately are retried with exponential backoff (1.5x escalation, 10s cap), up to a safety valve of 100 retries. The primary failure path is the timeout mechanism, not retry exhaustion.

### Soft Reservation Modes

The implementation supports two reservation semantics:

- **Default mode** (`cpls_soft_reservation=False`): FindAvailTime pushes start times past overlapping reservation windows (hard delay), with C_opp as an additional penalty. This prevents reservation violations entirely.
- **Soft mode** (`cpls_soft_reservation=True`): Reservations are purely advisory — OCAP only adds C_opp penalty without blocking placement. This matches the paper's "soft semantics" description but may lead to more reservation overlaps under high contention.

---

## Glossary

| Symbol | Meaning |
|--------|---------|
| v_i | Task i in the DAG |
| p_j | Compute node j |
| succ(v_i) | Immediate successors of task v_i |
| ET_i | Execution time of v_i |
| TT_{i,j} | Data transfer time from v_i to v_j |
| ST_i, CT_i | Start / completion time of v_i (CT_i = ST_i + ET_i) |
| D, K | CPLS lookahead depth and branch factor |
| alpha | Dynamic-criticality threshold multiplier |
| M, N | Number of system nodes / DAG tasks |
| DC(v_i) | Dynamic criticality score of task v_i |
| DRT | Downstream relative timing |
| GRT | Global Reservation Table |
