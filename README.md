# DREAM

**D**ynamic **R**ipple-**E**ffect-**A**ware **M**eta-Scheduling for Cloud-Edge-End Collaborative AI Computing

> Deterministic, explainable, and finite-time terminating DAG scheduling for heterogeneous cloud–edge–end networks. No offline training required.

---

## 📑 Table of Contents

- [What is DREAM?](#what-is-dream)
- [Key Features](#key-features)
- [Quick Start (API Sketch)](#quick-start-api-sketch)
- [System Architecture](#system-architecture)
- [Algorithm Design](#algorithm-design)
  - [Decision Flow](#decision-flow)
  - [CFG-to-DAG Preprocessing](#cfg-to-dag-preprocessing)
  - [CPLS — Critical-Path Lookahead Scheduler](#cpls--critical-path-lookahead-scheduler)
  - [OCAP — Opportunity-Cost-Aware Placement](#ocap--opportunity-cost-aware-placement)
  - [CPLS ↔ OCAP Coordination](#cpls--ocap-coordination)
- [Complexity & Guarantees](#complexity--guarantees)
- [Extensibility](#extensibility)
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
| 🔍 **Bounded lookahead** | CPLS explores up to $D$ hops along the highest-upward-rank successor chain, branching factor $K$, committing only the first hop. |
| 📊 **Dynamic criticality** | Tasks are classified at runtime via $\text{DC}(v_i) \geq \text{mean}(\text{DC}) + \alpha \cdot \text{std}(\text{DC})$, not by static critical path analysis. |
| 🛡️ **Soft reservations** | CPLS writes soft reservation windows for successor tasks; OCAP penalizes overlap via $C_{\mathrm{opp}}$ without hard exclusion, avoiding re-queue cascades. |
| ⚡ **Zero training at runtime** | Weights are optimized once offline by GA; online scheduling is fully deterministic and reproducible. |
| 🧩 **Plug-in extensibility** | New cost dimensions (energy, congestion, etc.) slot into OCAP as additional terms without altering CPLS or control flow. |
| ✅ **Finite-time guarantee** | Proven to terminate in at most $N$ rounds for a DAG with $N$ tasks; no backtracking, no asymptotic convergence tuning. |

---

## Quick Start (API Sketch)

Below is a **pseudocode** illustration of how DREAM is used in a target system. The actual repository will be published upon paper acceptance.

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

### Core assumptions

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
         b. For each feasible node compute 4-d surrogate cost
         c. Read reservation table; penalize overlap via C_opp
         d. Pick node with minimum total cost
4. Emit placement, update system state & reservation table
```

### CFG-to-DAG Preprocessing

Real AI workflows often arrive as control-flow graphs with loops and conditional branches. DREAM reduces them to DAGs once per workflow before online scheduling.

| Step | Action | Complexity |
|------|--------|------------|
| 1. SCC detection | Tarjan on the original graph | $O(\|V\| + \|F\|)$ |
| 2. Condensation | Collapse each SCC to a super-node | — |
| 3. Loop unrolling | Expand loops by iteration bound or probability weight | — |
| 4. Dependency rebuild | Preserve data and control semantics in the acyclic form | — |

> **Design note**: This is a pure preprocessing pass. The online scheduler never sees cycles.

### CPLS — Critical-Path Lookahead Scheduler

**Trigger**: $\text{DC}(v_i) \geq \text{mean}(\text{DC}) + \alpha \cdot \text{std}(\text{DC})$

#### Upward-rank successor selection

CPLS chooses which successor to follow via the upward rank:

$$
\text{rank}_u(v_i) = \overline{ET}_i + \max_{v_j \in \mathrm{succ}(v_i)} \left( \overline{TT}_{i,j} + \text{rank}_u(v_j) \right)
$$

This recursively captures the longest expected path from the current task to the exit node, ensuring the lookahead tree always follows the chain with the greatest impact on overall makespan.

#### Bounded search tree

For every critical task, CPLS grows a tree with:

- **Depth** $D$ — how many hops ahead to look.
- **Branch factor** $K$ — at each level expand the top-$K$ upward-rank successors.
- **Leaf evaluation** — estimate finish time at depth $D$ under a hypothetical system state.
- **First-hop commit** — backtrack the best leaf-to-root path, but **only the current task's placement is executed**; deeper positions are stored as **soft reservations**.

> **Why max-rank instead of top-$K$ branching per node?** Top-$K$ branching inflates cost from $O(D \cdot K^D \cdot M)$ to $O(D \cdot K^{D+1} \cdot M)$ per critical task, and probabilistic branching risks worst-case exponential blowup. Max-rank provides sufficient foresight in practice while keeping the bound tight.

**Default hyper-parameters**: $(D, K) = (4, 2)$.

### OCAP — Opportunity-Cost-Aware Placement

**Trigger**: $\text{DC}(v_i) < \text{mean}(\text{DC}) + \alpha \cdot \text{std}(\text{DC})$

#### Step 1 — Feasibility filtering (`FindAvailTime`)

OCAP first prunes the candidate node set by four hard constraints:

- **Dependency** — all predecessors finished and output data available.
- **Compute** — remaining compute capacity $\geq$ task execution time.
- **Memory** — remaining memory $\geq$ task peak footprint.
- **Model availability** — target node already caches (or can load) required model weights.

Only nodes passing all four filters proceed to cost scoring.

#### Step 2 — Four-dimensional surrogate cost

For each feasible node $p_j$:

$$
C(v_i, p_j) = w_{\mathrm{imm}} C_{\mathrm{imm}} + w_{\mathrm{rip}} C_{\mathrm{rip}} + w_{\mathrm{stab}} C_{\mathrm{stab}} + w_{\mathrm{opp}} C_{\mathrm{opp}}
$$

| Term | Computes | Purpose |
|------|----------|---------|
| $C_{\mathrm{imm}}$ | Expected finish time $ST_i + ET_i$ on $p_j$ | Avoid local waiting |
| $C_{\mathrm{rip}}$ | Aggregated DRT increment over $\mathrm{succ}(v_i)$ | Penalize placements that amplify downstream ripple |
| $C_{\mathrm{stab}}$ | Load-balance shift and fragmentation on $p_j$ | Prevent hotspot concentration |
| $C_{\mathrm{opp}}$ | Overlap area with CPLS-reserved critical windows | Protect critical chains without hard exclusion |

#### Step 3 — Weight vector & offline GA optimization

The scalarizing weights are fixed at runtime:

$$
(w_{\mathrm{imm}}, w_{\mathrm{rip}}, w_{\mathrm{stab}}, w_{\mathrm{opp}}) = (0.44, 0.27, 0.17, 0.12)
$$

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
| CFG-to-DAG reduction | $O(\|V\| + \|F\|)$ | Tarjan SCC; once per workflow |
| CPLS per critical task | $O(D \cdot K^D \cdot M)$ | Polynomial in $M$ when $D,K$ fixed |
| OCAP per non-critical task | $O(\|\mathrm{succ}(v_i)\| \cdot M^2)$ | Pairwise DRT evaluation |
| Amortized DAG bound | Eq. (overall_complexity) in paper | Full derivation in manuscript |

### Finite-time termination (Proposition 1)

DREAM is **finite-time terminating**, not asymptotically convergent:

1. **Progress** — every round commits at least one ready task.
2. **No backtracking** — committed placements are never undone.
3. **No revisits** — each task is processed exactly once.

Therefore an $N$-task DAG finishes in **at most $N$ scheduling rounds**. The offline GA weight optimization is a separate preprocessing cost and does not affect this online guarantee.

---

## Extensibility

DREAM's OCAP framework is designed for **structural plug-ins**: add a new cost term, run GA offline to learn its weight, and the online control flow stays identical.

| Extension | What changes | What stays the same |
|-----------|--------------|---------------------|
| **Time-varying bandwidth** | Replace piecewise-constant bandwidth with EWMA expected capacity; add congestion-history penalty to $C_{\mathrm{opp}}$ | CPLS/OCAP control flow, interfaces |
| **Energy-aware scheduling** | Add $C_{\mathrm{energy}} = \eta_j \cdot ET_i$; learn $w_{\mathrm{energy}}$ via GA | Normalization, algorithm skeleton, reservation mechanism |
| **Fault & retry** | Re-queue failed tasks; recompute DC under new resource landscape (may promote to CPLS) | Core scheduling loop, cost structure |
| **Joint energy–delay** | Multiple new cost terms in the same OCAP scalarizer | CPLS lookahead depth, soft reservation design |

---

## Glossary

| Symbol | Meaning |
|--------|---------|
| $v_i$ | Task $i$ in the DAG |
| $p_j$ | Compute node $j$ |
| $\mathrm{succ}(v_i)$ | Immediate successors of task $v_i$ |
| $ET_i$ | Execution time of $v_i$ |
| $TT_{i,j}$ | Data transfer time from $v_i$ to $v_j$ |
| $ST_i$, $CT_i$ | Start / completion time of $v_i$ ($CT_i = ST_i + ET_i$) |
| $D$, $K$ | CPLS lookahead depth and branch factor |
| $\alpha$ | Dynamic-criticality threshold multiplier |
| $M$, $N$ | Number of system nodes / DAG tasks |
| $\boldsymbol{st}$ | Start-time vector (bold-lowercase = vector, per IEEE convention) |
| $\boldsymbol{u'}$ | Hypothetical utilization vector |

---

## Citation

<!-- If you use DREAM in your research, please cite:

```bibtex
@article{dream2025tccn,
  title={DREAM: A Dynamic Ripple-Effect-Aware Meta-Scheduling Scheme for Cloud-Edge-End Collaborative AI Computing},
  author={Wang, Chenlu and Peng, Yuhuai and Liu, Lei and Sun, Geng and Dong, Mianxiong and Hu, Jiangang and Mumtaz, Shahid},
  journal={IEEE Transactions on Cloud Computing},
  year={2025}
}
``` -->

> **Note**: Source code and reproducibility scripts will be made publicly available upon paper acceptance.

