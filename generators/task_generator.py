# task_generator.py
import random
import networkx as nx
import json
import os
import copy
import math

# Directly in task_generator to avoid circular dependencies if models are needed elsewhere
def generate_model_versions(prefix, num_versions):
    return [f"{prefix}_v{i}" for i in range(1, num_versions + 1)]

def generate_all_models(small_versions=2, medium_versions=2, large_versions=1, super_large_versions=1):
    small_models = generate_model_versions("small", small_versions)
    medium_models = generate_model_versions("medium", medium_versions)
    large_models = generate_model_versions("large", large_versions)
    super_large_models = generate_model_versions("super_large", super_large_versions)
    return small_models + medium_models + large_models + super_large_models

ALL_MODELS_LIST = generate_all_models()

def get_duration_by_model(model_name):
    if model_name.startswith("small"): return random.uniform(0.1, 2.0)
    elif model_name.startswith("medium"): return random.uniform(2.1, 5.0)
    elif model_name.startswith("large"): return random.uniform(5.1, 10.0)
    elif model_name.startswith("super_large"): return random.uniform(10.1, 60.0)
    else: return 1.0

def get_compute_demand_by_model(model_name):
    if model_name.startswith("small"): return random.randint(100, 700)
    elif model_name.startswith("medium"): return random.randint(800, 2500)
    elif model_name.startswith("large"): return random.randint(3000, 8000)
    elif model_name.startswith("super_large"): 
        # 降低超大模型的上限，确保与云节点能力匹配
        # 云节点FLOPS能力: C1=353,160, C2=453,559, C3=366,481
        # 保留30%余量以防止资源竞争导致失败
        return random.randint(10000, 200000)
    else: return 500

class Task:
    id_counter = 0
    
    def __init__(self, task_type, compute_demand, memory_demand, data_size, model_required, original_id=None):
        Task.id_counter += 1
        self.id = f"T{Task.id_counter}"
        self.task_type = task_type
        self.compute_demand = compute_demand
        self.memory_demand = memory_demand
        self.data_size = data_size
        self.model_required = model_required
        self.duration = get_duration_by_model(model_required)
        self.original_id = original_id if original_id else self.id 

    def __str__(self):
        return f"{self.id} ({self.task_type})"

    def __repr__(self):
        return self.id
    
    @classmethod
    def reset_counter(cls):
        cls.id_counter = 0

def generate_task_dag(num_tasks, num_dags=1):
    Task.reset_counter()
    tasks = []
    for _ in range(num_tasks):
        model = random.choice(ALL_MODELS_LIST)
        task_type = "" 
        mem_min, mem_max, dmin, dmax = 0,0,0,0 

        if model.startswith("small"): 
            task_type = "image_processing"
            mem_min, mem_max, dmin, dmax = 4, 32, 100, 500
        elif model.startswith("medium"): 
            task_type = random.choice(["image_processing", "nlp"])
            if "image" in task_type: 
                mem_min, mem_max, dmin, dmax = 4, 32, 100, 500
            else: 
                mem_min, mem_max, dmin, dmax = 16, 128, 10, 100
        elif model.startswith("large"): 
            task_type = random.choice(["nlp", "recommendation"])
            if "nlp" in task_type: 
                mem_min, mem_max, dmin, dmax = 16, 128, 10, 100
            else: 
                mem_min, mem_max, dmin, dmax = 64, 256, 500, 2000
        else: 
            task_type = "recommendation"
            mem_min, mem_max, dmin, dmax = 64, 256, 500, 2000
        
        tasks.append(Task(task_type, get_compute_demand_by_model(model), random.randint(mem_min, mem_max), random.randint(dmin, dmax), model))
    
    G = nx.DiGraph()
    for task_node in tasks: G.add_node(task_node)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if random.random() < 0.3: 
                G.add_edge(tasks[i], tasks[j])
    
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            if cycle: 
                 G.remove_edge(cycle[0][0], cycle[0][1])
            else: 
                break 
        except nx.NetworkXNoCycle:
            break
    if num_dags == 1: return G
    return [G] 

def _visualize_single_dag(G, filename, scc_graph=False):
    import matplotlib.pyplot as plt  # Lazy import for visualization only
    # 如果filename为None或空字符串，则不执行任何可视化操作
    if filename is None or filename == "":
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        # 如果输出目录不存在且不是空字符串，则直接返回，避免创建目录
        return
    
    # Skip visualization for large graphs to save memory
    if G.number_of_nodes() > 100:
        print(f"Skipping visualization for large graph with {G.number_of_nodes()} nodes to save memory")
        # Just save an empty image with a message
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Graph too large to visualize: {G.number_of_nodes()} nodes",
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.savefig(filename, dpi=72)  # Lower DPI to save memory
        plt.close()
        return
    
    if G.number_of_nodes() == 0: return
    
    # Use smaller figure size for larger graphs
    if G.number_of_nodes() > 50:
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(12, 10))
    
    # Layout calculation with memory optimization
    if scc_graph:
        pos = nx.kamada_kawai_layout(G) # Better for SCC condensation graph
    elif G.number_of_nodes() <= 20:
        pos = nx.spring_layout(G, k=0.9, seed=42)
    else:
        # For larger graphs, use faster layout algorithm
        pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)  # Fewer iterations to save memory
    
    # Draw nodes with size adaptation
    if scc_graph:
        nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.8, node_color='lightblue')
        
        # Label SCC nodes 
        labels = {}
        for node in G.nodes():
            labels[node] = node
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    else: 
        # Task nodes - color by task type but with simplified logic
        node_colors = []
        for node in G.nodes():
            task_type = getattr(node, 'task_type', 'unknown')
            if 'image' in task_type: node_colors.append('skyblue')
            elif 'nlp' in task_type: node_colors.append('lightgreen')
            elif 'recommendation' in task_type: node_colors.append('lightcoral')
            else: node_colors.append('gray')
        
        # Adjust node size based on graph size (more aggressive scaling)
        node_size = max(100, 1000 - G.number_of_nodes() * 5)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.8, node_color=node_colors)
        
        # For larger graphs, only label a subset of nodes to reduce clutter
        labels = {}
        if G.number_of_nodes() > 50:
            # Just label every 5th node
            nodes_list = list(G.nodes())
            for i, node in enumerate(nodes_list):
                if i % 5 == 0:  # Only label every 5th node
                    labels[node] = getattr(node, 'id', str(node))
        else:
            for node in G.nodes():
                labels[node] = getattr(node, 'id', str(node))
        
        # Adaptive font size
        font_size = max(4, min(8, 10 - G.number_of_nodes() // 15))
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)
    
    # Draw edges with arrows - simplified for larger graphs
    if G.number_of_nodes() > 50:
        # Simpler edge rendering for large graphs
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=0.5, alpha=0.4)
    else:
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, arrowstyle='->', width=1.0, alpha=0.6)
    
    plt.title(os.path.basename(filename).replace('.png', ''))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=80)  # Reduced DPI to save memory
    
    # 导出DAG图表数据
    try:
        tasks_data = []
        for task_id in G.nodes():
            task_data = G.nodes[task_id]
            tasks_data.append({
                "task_id": str(task_id),
                "compute_demand": task_data.get("compute_demand", 0),
                "memory_demand": task_data.get("memory_demand", 0),
                "models": task_data.get("models", []),
                "predecessors": [str(pred) for pred in G.predecessors(task_id)],
                "successors": [str(succ) for succ in G.successors(task_id)],
                "position": pos.get(task_id, [0, 0]) if pos else [0, 0],
                **{k: v for k, v in task_data.items() if k not in ["compute_demand", "memory_demand", "models"]}
            })
        
        # Convert position dictionary keys to strings for JSON serialization
        layout_positions = {}
        if pos:
            for node, position in pos.items():
                layout_positions[str(node)] = position
        
        dag_data = {
            "dag_type": "task_dag" if not scc_graph else "scc_overview",
            "total_tasks": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "tasks": tasks_data,
            "layout_positions": layout_positions
        }
        
    except Exception as e:
        print(f"Warning: Could not export DAG chart data for {filename}: {e}")
    
    import gc
    gc.collect()

def visualize_task_dag(G_or_list_of_G, base_filename_no_ext):
    if isinstance(G_or_list_of_G, list):
        for i, dag_g in enumerate(G_or_list_of_G):
             _visualize_single_dag(dag_g, f"{base_filename_no_ext}_{i+1}.png")
    elif isinstance(G_or_list_of_G, nx.DiGraph):
         _visualize_single_dag(G_or_list_of_G, f"{base_filename_no_ext}.png", verbose=verbose)
    else:
        if verbose:
            print(f"Warning: Could not visualize type {type(G_or_list_of_G)}")


def _generate_cyclic_graph(tasks_definitions, edge_prob: float = 0.15, verbose: bool = True):
    G = nx.DiGraph()
    task_nodes = []

    # For very large task counts, batch the node creation to save memory
    batch_size = 100
    for i, (task_type, compute, mem, data_size, model) in enumerate(tasks_definitions):
        task_node = Task(task_type, compute, mem, data_size, model, original_id=f"OrigT{i+1}")
        task_node.id = task_node.original_id 
        task_nodes.append(task_node)
        G.add_node(task_node)
        
        # Force garbage collection periodically during large task creation
        if i > 0 and i % batch_size == 0:
            import gc
            gc.collect()

    # Reduce edge density for larger graphs to prevent memory explosion
    n = len(task_nodes)
    
    # 修复：保持合理的边概率以维持DAG连通性和并行度
    # 避免过度稀疏化导致makespan随节点数增加而上升
    if n <= 50:
        # 小规模：轻微调整
        adaptive_edge_prob = edge_prob * (0.8 + 0.2 * (50 - n) / 50)
    else:
        # 大规模：保持最小边概率以确保连通性和并行度
        # 使用更温和的缩放，确保有足够的边来维持并行性
        min_edge_prob = 0.05  # 最小边概率阈值
        scale_factor = max(0.3, 50.0 / n)  # 温和的缩放因子
        adaptive_edge_prob = max(min_edge_prob, edge_prob * scale_factor)
    
    if verbose:
        print(f"Using adaptive edge probability: {adaptive_edge_prob:.6f} for {n} nodes")
    
    # 🔥 关键修复：重新设计DAG生成以确保真正的并行性
    # 问题：之前的方法生成完全串行的链式DAG（最大并行度=1）
    # 解决方案：使用分层并行结构，确保每层有多个可并行执行的任务
    
    if verbose:
        print(f"生成具有并行性的分层DAG结构...")
    
    # 创建分层并行DAG结构
    # 目标：确保平均并行度 > 1，关键路径长度 < 总任务数的50%
    target_layers = max(3, min(int(math.sqrt(n)), n // 3))  # 合理的层数
    tasks_per_layer = n // target_layers  # 每层的平均任务数
    
    # 按层分配任务
    layers = []
    for layer_idx in range(target_layers):
        start_idx = layer_idx * tasks_per_layer
        end_idx = min((layer_idx + 1) * tasks_per_layer, n)
        if layer_idx == target_layers - 1:  # 最后一层包含所有剩余任务
            end_idx = n
        layers.append(list(range(start_idx, end_idx)))
    
    if verbose:
        layer_sizes = [len(layer) for layer in layers]
        print(f"创建{len(layers)}层DAG，每层任务数: {layer_sizes}")
    
    # 层内连接：同层内的任务可以并行执行，添加少量依赖关系
    for layer_tasks in layers:
        if len(layer_tasks) > 1:
            # 在同层内创建少量依赖（保持大部分并行性）
            dependency_prob = 0.2  # 只有20%的概率创建同层依赖
            for i in range(len(layer_tasks) - 1):
                if random.random() < dependency_prob:
                    src_idx = layer_tasks[i]
                    tgt_idx = layer_tasks[i + 1]
                    G.add_edge(task_nodes[src_idx], task_nodes[tgt_idx])
    
    # 层间连接：确保前一层的任务连接到后一层
    for layer_idx in range(len(layers) - 1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        
        # 每个后层任务从前层选择1-3个依赖
        for next_task_idx in next_layer:
            num_dependencies = min(len(current_layer), random.randint(1, 3))
            dependencies = random.sample(current_layer, num_dependencies)
            
            for dep_task_idx in dependencies:
                G.add_edge(task_nodes[dep_task_idx], task_nodes[next_task_idx])
    
    # 验证并行性
    if nx.is_directed_acyclic_graph(G):
        # 计算理论并行度
        try:
            longest_path_length = nx.dag_longest_path_length(G)
            theoretical_parallelism = n / longest_path_length if longest_path_length > 0 else 1
            if verbose:
                print(f"DAG并行性验证: 关键路径={longest_path_length}, 理论并行度={theoretical_parallelism:.1f}")
        except:
            if verbose:
                print("无法计算DAG并行性指标")
    
    # Create a very limited number of small cycles
    if n >= 3 and (nx.is_directed_acyclic_graph(G) or random.random() < 0.3):  # 30% chance to add more cycles
        # Limit cycles based on graph size
        num_cycles = min(3, max(1, n // 50))
        if verbose:
            print(f"Adding {num_cycles} small cycles to the graph")
        for _ in range(num_cycles):
            if n < 3: break
            # Create only small 3-node cycles
            potential_cycle_nodes = random.sample(task_nodes, 3)
            a, b, c = potential_cycle_nodes
            if not G.has_edge(a, b): G.add_edge(a, b)
            if not G.has_edge(b, c): G.add_edge(b, c)
            if not G.has_edge(c, a): G.add_edge(c, a)
            
    if verbose:
        print(f"Generated graph with {n} nodes and {G.number_of_edges()} edges")
    return G


def _build_graph_nodes_from_definitions(tasks_definitions):
    G = nx.DiGraph()
    task_nodes = []
    for i, (task_type, compute, mem, data_size, model) in enumerate(tasks_definitions):
        task_node = Task(task_type, compute, mem, data_size, model, original_id=f"OrigT{i+1}")
        task_node.id = task_node.original_id
        task_nodes.append(task_node)
        G.add_node(task_node)
    return G, task_nodes


def _generate_wide_graph(tasks_definitions, verbose: bool = True):
    G, task_nodes = _build_graph_nodes_from_definitions(tasks_definitions)
    n = len(task_nodes)
    if n <= 1:
        return G

    layer_count = max(3, min(6, n // 20 + 3))
    layer_count = min(layer_count, n)
    base_layer_size = n // layer_count
    remainder = n % layer_count

    layers = []
    cursor = 0
    for layer_idx in range(layer_count):
        layer_size = base_layer_size + (1 if layer_idx < remainder else 0)
        if layer_size <= 0:
            continue
        layer_nodes = task_nodes[cursor : cursor + layer_size]
        layers.append(layer_nodes)
        cursor += layer_size

    for layer_idx in range(len(layers) - 1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        for node in next_layer:
            dep_count = min(len(current_layer), random.randint(1, min(4, len(current_layer))))
            for pred in random.sample(current_layer, dep_count):
                G.add_edge(pred, node)

        if layer_idx + 2 < len(layers):
            skip_layer = layers[layer_idx + 2]
            for src in current_layer:
                if random.random() < 0.25:
                    dst = random.choice(skip_layer)
                    G.add_edge(src, dst)

    if verbose:
        print(f"Generated wide topology graph with {n} nodes, {len(layers)} layers and {G.number_of_edges()} edges")
    return G


def _generate_deep_graph(tasks_definitions, verbose: bool = True):
    """Generate a deep DAG topology with multiple parallel chains, cross-chain sync points,
    and diamond (fan-out/fan-in) patterns.
    
    Creates `num_chains` parallel chains of sequential tasks with:
    1. Periodic cross-chain sync/merge nodes (fan-in from multiple chains)
    2. Diamond patterns where a sync node fans out to multiple chains again
    3. Random cross-chain edges for additional scheduling freedom
    
    This preserves the "deep" character (long critical paths) while giving
    scheduling algorithms meaningful choices (which chain to prioritize,
    where to place sync/fork tasks on high-capability nodes, how to
    balance critical path length vs resource utilization).
    """
    G, task_nodes = _build_graph_nodes_from_definitions(tasks_definitions)
    n = len(task_nodes)
    if n <= 1:
        return G

    # More chains for richer scheduling choices (4-8 depending on task count)
    num_chains = max(4, min(8, n // 40))
    
    # Distribute tasks across chains
    chains = [[] for _ in range(num_chains)]
    for i, node in enumerate(task_nodes):
        chains[i % num_chains].append(node)
    
    # Build sequential edges within each chain
    for chain in chains:
        for i in range(len(chain) - 1):
            G.add_edge(chain[i], chain[i + 1])
    
    # Add cross-chain sync/merge points more frequently (every ~8-12 tasks)
    sync_interval = max(8, n // (num_chains * 5))
    for chain_idx, chain in enumerate(chains):
        for step in range(sync_interval, len(chain), sync_interval):
            if step < len(chain):
                # Fan-in: connect from 2-3 adjacent chains to this sync node
                num_incoming = min(3, num_chains - 1)
                incoming_chains = [(chain_idx + d) % num_chains for d in range(1, num_incoming + 1)]
                for other_chain_idx in incoming_chains:
                    other_chain = chains[other_chain_idx]
                    other_step = min(step, len(other_chain) - 1)
                    if other_step > 0:
                        G.add_edge(other_chain[other_step - 1], chain[step])
                
                # Fan-out: from this sync node, add edges to next segment of adjacent chains
                if step + sync_interval < len(chain):
                    for other_chain_idx in incoming_chains:
                        other_chain = chains[other_chain_idx]
                        target_step = min(step + sync_interval, len(other_chain) - 1)
                        if target_step > step:
                            G.add_edge(chain[step], other_chain[target_step])
    
    # Add random cross-chain edges for additional scheduling freedom
    for chain_idx, chain in enumerate(chains):
        for i in range(len(chain)):
            if random.random() < 0.06:
                other_chain_idx = random.choice([j for j in range(num_chains) if j != chain_idx])
                other_chain = chains[other_chain_idx]
                # Connect to a nearby position in the other chain
                offset = random.randint(-3, 3)
                other_pos = max(0, min(len(other_chain) - 1, i + offset))
                if other_pos > 0:
                    G.add_edge(other_chain[other_pos - 1], chain[i])
    
    # Add some skip edges for additional depth variation
    for chain in chains:
        for i in range(len(chain) - 2):
            if random.random() < 0.08:
                jump = min(len(chain) - 1, i + random.randint(2, min(4, len(chain) - i - 1)))
                if jump > i + 1:
                    G.add_edge(chain[i], chain[jump])

    if verbose:
        print(f"Generated deep topology graph with {n} nodes, {num_chains} chains and {G.number_of_edges()} edges")
    return G


def _generate_topology_seed_graph(tasks_definitions, topology_type: str = "random", verbose: bool = True):
    topology = (topology_type or "random").lower()
    # Support continuous depth parameter: "depth_0.3" means 30% deep / 70% wide
    if topology.startswith("depth_"):
        try:
            depth_ratio = float(topology.split("_", 1)[1])
            depth_ratio = max(0.0, min(1.0, depth_ratio))
            return _generate_mixed_graph(tasks_definitions, depth_ratio, verbose=verbose)
        except (ValueError, IndexError):
            pass
    if topology == "wide":
        return _generate_wide_graph(tasks_definitions, verbose=verbose)
    if topology == "deep":
        return _generate_deep_graph(tasks_definitions, verbose=verbose)
    return _generate_cyclic_graph(tasks_definitions, verbose=verbose)


def _generate_mixed_graph(tasks_definitions, depth_ratio: float, verbose: bool = True):
    """Generate a DAG topology that interpolates between wide (0.0) and deep (1.0).

    depth_ratio controls the structure:
      - 0.0: pure wide (many parallel branches, few layers)
      - 0.5: balanced (moderate depth and width)
      - 1.0: pure deep (few long chains with sync points)

    The interpolation works by controlling:
      1. Number of chains (more chains = wider)
      2. Chain length (longer chains = deeper)
      3. Cross-chain edge density (more cross-edges = wider)
      4. Skip-edge probability (more skip-edges = deeper shortcuts)
    """
    G, task_nodes = _build_graph_nodes_from_definitions(tasks_definitions)
    n = len(task_nodes)
    if n <= 1:
        return G

    # At depth_ratio=0 (wide): many short chains, high cross-connectivity
    # At depth_ratio=1 (deep): few long chains, low cross-connectivity
    # Number of chains: wide → many, deep → few
    wide_chains = max(8, n // 10)   # wide: ~10 tasks per chain
    deep_chains = max(4, n // 60)   # deep: ~60 tasks per chain
    num_chains = max(2, int(wide_chains * (1 - depth_ratio) + deep_chains * depth_ratio))

    # Distribute tasks across chains
    chains = [[] for _ in range(num_chains)]
    for i, node in enumerate(task_nodes):
        chains[i % num_chains].append(node)

    # Build sequential edges within each chain
    for chain in chains:
        for i in range(len(chain) - 1):
            G.add_edge(chain[i], chain[i + 1])

    # Cross-chain connectivity: high at depth=0, low at depth=1
    # Wide: every node gets cross-chain edges with moderate probability
    # Deep: only periodic sync points
    cross_edge_prob = 0.3 * (1 - depth_ratio)  # 0.3 at wide, 0.0 at deep
    sync_interval = max(5, int(20 * depth_ratio + 5 * (1 - depth_ratio)))  # 5 at wide, 25 at deep

    for chain_idx, chain in enumerate(chains):
        # Periodic sync/merge points (always present, but frequency varies)
        for step in range(sync_interval, len(chain), sync_interval):
            if step < len(chain):
                # Number of incoming chains varies with depth
                num_incoming = max(1, int(3 * (1 - depth_ratio * 0.5)))
                num_incoming = min(num_incoming, num_chains - 1)
                incoming_chains = [(chain_idx + d) % num_chains for d in range(1, num_incoming + 1)]
                for other_chain_idx in incoming_chains:
                    other_chain = chains[other_chain_idx]
                    other_step = min(step, len(other_chain) - 1)
                    if other_step > 0:
                        G.add_edge(other_chain[other_step - 1], chain[step])

                # Fan-out from sync nodes (less at deep end)
                if depth_ratio < 0.7 and step + sync_interval < len(chain):
                    for other_chain_idx in incoming_chains[:1]:
                        other_chain = chains[other_chain_idx]
                        target_step = min(step + sync_interval, len(other_chain) - 1)
                        if target_step > step:
                            G.add_edge(chain[step], other_chain[target_step])

        # Random cross-chain edges (wide topology style)
        if cross_edge_prob > 0:
            for i in range(len(chain)):
                if random.random() < cross_edge_prob:
                    other_chain_idx = random.choice([j for j in range(num_chains) if j != chain_idx])
                    other_chain = chains[other_chain_idx]
                    offset = random.randint(-3, 3)
                    other_pos = max(0, min(len(other_chain) - 1, i + offset))
                    if other_pos > 0:
                        G.add_edge(other_chain[other_pos - 1], chain[i])

    # Skip edges (deep topology style — more at deep end)
    skip_prob = 0.08 * depth_ratio  # 0 at wide, 0.08 at deep
    if skip_prob > 0:
        for chain in chains:
            for i in range(len(chain) - 2):
                if random.random() < skip_prob:
                    jump = min(len(chain) - 1, i + random.randint(2, min(4, len(chain) - i - 1)))
                    if jump > i + 1:
                        G.add_edge(chain[i], chain[jump])

    if verbose:
        print(f"Generated mixed topology (depth={depth_ratio:.2f}) with {n} nodes, "
              f"{num_chains} chains and {G.number_of_edges()} edges")
    return G

def _unfold_scc(scc_subgraph: nx.DiGraph, super_node_id_prefix: str, max_k: int, task_id_counter_obj: type[Task], verbose: bool = True): 
    dag = nx.DiGraph()
    first_map = {}  
    last_map = {}   
    
    original_nodes_in_scc = sorted(list(scc_subgraph.nodes()), key=lambda n: getattr(n, 'original_id', n.id))

    if not original_nodes_in_scc: return dag, first_map, last_map

    is_trivial_scc = (scc_subgraph.number_of_nodes() == 1 and scc_subgraph.number_of_edges() == 0)
    
    # Adaptive K calculation based on SCC size to limit memory usage
    scc_size = len(original_nodes_in_scc)
    if is_trivial_scc:
        K = 1
    else:
        # Scale max_k down for larger SCCs
        adjusted_max_k = max(2, min(max_k, int(10 / (scc_size/5 + 1))))
        K = random.randint(2, adjusted_max_k)

    copies_map = {} 
    for k_iter in range(1, K + 1):
        for i, orig_node in enumerate(original_nodes_in_scc):
            new_task_instance = copy.deepcopy(orig_node)
            task_id_counter_obj.id_counter +=1 
            new_task_instance.id = f"T{task_id_counter_obj.id_counter}"
            new_task_instance.original_id = f"{super_node_id_prefix}-{orig_node.id.replace('OrigT','')}-{k_iter}" 
            
            dag.add_node(new_task_instance)
            copies_map[(orig_node, k_iter)] = new_task_instance
            
            if k_iter == 1:
                first_map[orig_node] = new_task_instance
            if k_iter == K:
                last_map[orig_node] = new_task_instance

    if K > 1: 
        for k_iter in range(1, K): 
            for u_orig, v_orig in scc_subgraph.edges():
                if (u_orig, k_iter) in copies_map and (v_orig, k_iter + 1) in copies_map:
                    dag.add_edge(copies_map[(u_orig, k_iter)], copies_map[(v_orig, k_iter + 1)])
    
    if K == 1 and original_nodes_in_scc: 
        orig_node = original_nodes_in_scc[0]
        if (orig_node, 1) in copies_map: 
            single_copy = copies_map[(orig_node, 1)]
            first_map[orig_node] = single_copy
            last_map[orig_node] = single_copy
        
    return dag, first_map, last_map


def generate_dynamic_task_dag(num_initial_tasks: int = 30, 
                              max_unfold_iterations: int = 3, 
                              base_filename_prefix="T0",
                              output_visualization_dir=None,
                              dag_topology: str = "random",
                              verbose: bool = True):
    Task.reset_counter() # Reset for G_raw's temp original_ids (if Task class used directly by _generate_cyclic_graph)
    
    # For very large graphs, use more aggressive optimizations
    adjusted_initial_tasks = num_initial_tasks
    if verbose:
        print(f"Generating {adjusted_initial_tasks} initial tasks for the DAG")
    
    # Keep max_unfold constant across all task counts to eliminate performance jumps
    # Only adjust SCC processing parameters smoothly
    adjusted_max_unfold = max_unfold_iterations  # No adjustment - keep constant!
    
    if num_initial_tasks <= 30:
        # Small task counts: use original SCC parameters
        scc_size_divisor = 4
        scc_count_divisor = 5
    else:
        # For larger task counts, smoothly adjust only the SCC divisors
        # Calculate progress factor from 0.0 (at 30 tasks) to 1.0 (at 200+ tasks)
        progress_factor = min(1.0, (num_initial_tasks - 30) / 170.0)
        
        # Smoothly adjust divisors only - keep max_unfold unchanged
        scc_size_divisor = max(1, int(4 - 3 * progress_factor))
        scc_count_divisor = max(2, int(5 - 3 * progress_factor))
    
    task_definitions_for_g_raw = []
    for _ in range(adjusted_initial_tasks):
        model = random.choice(ALL_MODELS_LIST)
        task_type = "" 
        mem_min, mem_max, dmin, dmax = 0,0,0,0 
        if model.startswith("small"): 
            task_type = "image_processing"
            mem_min, mem_max, dmin, dmax = 4, 32, 100, 500
        elif model.startswith("medium"): 
            task_type = random.choice(["image_processing", "nlp"])
            if "image" in task_type: 
                mem_min, mem_max, dmin, dmax = 4, 32, 100, 500
            else: 
                mem_min, mem_max, dmin, dmax = 16, 128, 10, 100
        elif model.startswith("large"): 
            task_type = random.choice(["nlp", "recommendation"])
            if "nlp" in task_type: 
                mem_min, mem_max, dmin, dmax = 16, 128, 10, 100
            else: 
                mem_min, mem_max, dmin, dmax = 64, 256, 500, 2000
        else: 
            task_type = "recommendation"
            mem_min, mem_max, dmin, dmax = 64, 256, 500, 2000
        task_definitions_for_g_raw.append((task_type, get_compute_demand_by_model(model), random.randint(mem_min, mem_max), random.randint(dmin, dmax), model))

    G_raw = _generate_topology_seed_graph(task_definitions_for_g_raw, topology_type=dag_topology, verbose=verbose)
    
    # 只在output_visualization_dir有效时执行可视化
    if output_visualization_dir is not None:
        _visualize_single_dag(G_raw, os.path.join(output_visualization_dir, f"{base_filename_prefix}_G_raw_initial_graph.png"))

    # Memory optimization: Limit SCC complexity
    sccs = list(nx.strongly_connected_components(G_raw))
    
    # Improved SCC filtering logic: use smooth scaling functions to avoid performance jumps
    # Calculate scaling factors based on task count using continuous functions
    # Use a larger range to avoid sudden stops at 100 tasks
    scale_factor = min(1.0, num_initial_tasks / 200.0)  # Scale from 0 to 1 for 0-200 tasks
    
    # Use more conservative SCC limits that scale smoothly
    # Base limits that work well for small task counts
    base_min_sccs = 20
    base_max_sccs = 50
    
    # Scale limits smoothly based on task count using continuous values
    # Use floating point calculations and round only at the final step to minimize discrete jumps
    min_sccs_float = max(base_min_sccs, base_min_sccs + scale_factor * num_initial_tasks * 0.3)
    max_sccs_float = max(min_sccs_float, base_max_sccs + scale_factor * num_initial_tasks * 0.5)
    
    # Round to integers but with smoother transitions
    min_sccs = int(round(min_sccs_float))
    max_sccs = int(round(max_sccs_float))
    
    # Only filter if we have significantly more SCCs than the limit
    # Use a buffer zone to avoid filtering at borderline cases
    filter_threshold = max_sccs * 1.2  # Only filter if 20% above limit
    
    if len(sccs) > filter_threshold:
        if verbose:
            print(f"Applying smooth SCC filtering: {len(sccs)} -> {max_sccs} SCCs (task_count={num_initial_tasks})")
        
        # Sort SCCs by size, keep the largest ones
        sccs = sorted(sccs, key=len, reverse=True)
        
        # Use a more generous node retention target (95% instead of 90%)
        # This preserves more of the original DAG structure
        target_node_ratio = 0.95
        target_nodes = int(num_initial_tasks * target_node_ratio)
        
        kept_sccs = []
        kept_nodes = set()
        
        # Keep SCCs until we reach target nodes and minimum SCC count
        for scc in sccs:
            # Always keep enough SCCs to meet minimum requirement
            if len(kept_sccs) < min_sccs:
                kept_sccs.append(scc)
                kept_nodes.update(scc)
            # After minimum, only add if we haven't reached targets
            elif len(kept_nodes) < target_nodes and len(kept_sccs) < max_sccs:
                kept_sccs.append(scc)
                kept_nodes.update(scc)
            else:
                break
        
        if verbose:
            retention_ratio = len(kept_nodes) / num_initial_tasks
            print(f"SCC filtering result: {len(kept_nodes)} nodes ({retention_ratio:.1%}) from {len(kept_sccs)} SCCs")
        
        sccs = kept_sccs
        G_raw = G_raw.subgraph(kept_nodes).copy()
    else:
        if verbose:
            print(f"No SCC filtering needed: {len(sccs)} SCCs <= {filter_threshold:.0f} threshold")
    
    scc_node_map = {}
    for i, scc_member_set in enumerate(sccs):
        for node in scc_member_set:
            scc_node_map[node] = i 
    
    G_condensed_labeled = nx.DiGraph() 
    if G_raw.number_of_nodes() > 0: 
        G_condensed = nx.condensation(G_raw, sccs) # G_condensed nodes are integers
        node_relabel_map = {}
        for node_idx_in_condensation_graph in G_condensed.nodes(): # These are original SCC indices
            super_node_label = f"{base_filename_prefix}-SCC{node_idx_in_condensation_graph}"
            node_relabel_map[node_idx_in_condensation_graph] = super_node_label
            G_condensed_labeled.add_node(super_node_label)

        for u_idx, v_idx, data in G_condensed.edges(data=True):
             if u_idx in node_relabel_map and v_idx in node_relabel_map: 
                G_condensed_labeled.add_edge(node_relabel_map[u_idx], node_relabel_map[v_idx], **data)
    
    # 只在output_visualization_dir有效时执行可视化
    if output_visualization_dir is not None:
        _visualize_single_dag(G_condensed_labeled, os.path.join(output_visualization_dir, f"{base_filename_prefix}_SCC_overview.png"), scc_graph=True)

    final_dag = nx.DiGraph()
    map_orig_to_first_unfolded = {} 
    map_orig_to_last_unfolded = {}

    Task.reset_counter() # CRITICAL: Reset counter *before* generating tasks for the final DAG

    for i, scc_member_set in enumerate(sccs):
        if not scc_member_set: continue
        
        # Improved SCC size management: use more generous and smooth limits
        # Scale maximum SCC size based on total task count to avoid sudden restrictions
        # Use consistent 200-task range to avoid sudden stops at 100 tasks
        scale_factor = min(1.0, num_initial_tasks / 200.0)
        
        # More generous size limits that scale with task count using continuous calculations
        min_scc_size_float = max(5.0, 5.0 + scale_factor * 10.0)  # 5-15 range
        max_scc_size_float = max(50.0, 50.0 + scale_factor * num_initial_tasks * 0.3)  # More generous limit
        
        # Convert to integers with smooth rounding
        min_scc_size = int(round(min_scc_size_float))
        max_scc_size = int(round(max_scc_size_float))
        
        current_size = len(scc_member_set)
        
        # Only limit very large SCCs (above 150% of the target)
        limit_threshold = int(max_scc_size * 1.5)
        
        if current_size > limit_threshold:
            # Preserve more of the original structure - keep 85% instead of 80%
            target_size = max(max_scc_size, int(current_size * 0.85))
            
            # Use deterministic selection to preserve structure consistency
            sorted_nodes = sorted(list(scc_member_set), key=lambda x: int(x.id[1:]) if x.id[1:].isdigit() else 0)
            scc_member_set = set(sorted_nodes[:target_size])
            
            if verbose:
                print(f"SCC{i}: Reduced from {current_size} to {target_size} nodes (preserved {target_size/current_size:.1%})")
            
        scc_subgraph = G_raw.subgraph(scc_member_set).copy()
        super_node_id_for_unfolding = f"{base_filename_prefix}-SCC{i}"
        
        # 只在output_visualization_dir有效时执行可视化
        if output_visualization_dir is not None:
            _visualize_single_dag(scc_subgraph, os.path.join(output_visualization_dir, f"{super_node_id_for_unfolding}_internal_structure.png"))
        
        unfolded_scc_dag, first_map, last_map = _unfold_scc(scc_subgraph, super_node_id_for_unfolding, adjusted_max_unfold, Task)
        
        final_dag = nx.compose(final_dag, unfolded_scc_dag) 
        map_orig_to_first_unfolded.update(first_map)
        map_orig_to_last_unfolded.update(last_map)

    # Memory optimization: Adaptive limit for edges between SCCs based on task count
    max_inter_scc_edges = max(50, min(700, num_initial_tasks * 2))
    inter_scc_edges = []
    for u_orig, v_orig in G_raw.edges():
        if (u_orig in scc_node_map and v_orig in scc_node_map and 
            scc_node_map[u_orig] != scc_node_map[v_orig] and
            u_orig in map_orig_to_last_unfolded and v_orig in map_orig_to_first_unfolded):
            inter_scc_edges.append((u_orig, v_orig))
    
    # Randomly sample edges if we have too many
    if len(inter_scc_edges) > max_inter_scc_edges:
        print(f"Warning: Limiting inter-SCC edges from {len(inter_scc_edges)} to {max_inter_scc_edges} to prevent memory issues")
        inter_scc_edges = random.sample(inter_scc_edges, max_inter_scc_edges)
    
    # Add the selected edges to the final DAG
    for u_orig, v_orig in inter_scc_edges:
        last_u_instance = map_orig_to_last_unfolded[u_orig]
        first_v_instance = map_orig_to_first_unfolded[v_orig]
        if not final_dag.has_edge(last_u_instance, first_v_instance):
            final_dag.add_edge(last_u_instance, first_v_instance)

    if not nx.is_weakly_connected(final_dag) and final_dag.number_of_nodes() > 1:
        if verbose:
            print(f"修复DAG连通性：检测到{final_dag.number_of_nodes()}个节点的DAG不连通，正在修复...")
        
        # 获取所有连通分量
        components = list(nx.weakly_connected_components(final_dag))
        if verbose:
            print(f"发现{len(components)}个连通分量，大小分别为: {[len(c) for c in components]}")
        
        # 如果有多个连通分量，在它们之间添加边以确保连通性
        if len(components) > 1:
            # 按组件大小排序
            components = sorted(components, key=len, reverse=True)
            
            # 连接相邻的连通分量
            for i in range(len(components) - 1):
                # 从较大组件选择一个节点作为源
                source_component = components[i]
                target_component = components[i + 1]
                
                # 选择度数较小的节点来减少对DAG结构的影响
                source_node = min(source_component, key=lambda n: final_dag.out_degree(n))
                target_node = min(target_component, key=lambda n: final_dag.in_degree(n))
                
                # 添加连接边
                if not final_dag.has_edge(source_node, target_node):
                    final_dag.add_edge(source_node, target_node)
                    if verbose:
                        print(f"  添加连接边: {source_node.original_id} -> {target_node.original_id}")
        
        # 验证修复结果
        if nx.is_weakly_connected(final_dag):
            if verbose:
                print(f"✓ DAG连通性修复成功！现在DAG是连通的。")
        else:
            if verbose:
                print(f"✗ DAG连通性修复失败，仍然不连通。")

    if final_dag.number_of_nodes() > 0: 
        assert nx.is_directed_acyclic_graph(final_dag), "Final unfolded graph is not a DAG!"
        # Visualization of the final DAG is handled by the caller (simulation_main.py)
    else:
        if verbose:
            print("Warning: Final dynamic DAG is empty.")
        
    return final_dag


def task_dag_to_json(task_dag_g):
    tasks_list = []
    if not isinstance(task_dag_g, nx.DiGraph): 
        return {"tasks": [], "dependencies": []}
        
    for t_node in task_dag_g.nodes():
        tasks_list.append({
            "id": t_node.id,
            "type": t_node.task_type,
            "compute_demand": t_node.compute_demand,
            "memory_demand": t_node.memory_demand,
            "data_size": t_node.data_size,
            "model_required": t_node.model_required,
            "duration": round(t_node.duration, 2),
            "original_id": getattr(t_node, 'original_id', t_node.id) 
        })
    
    dependencies_list = []
    for src_t, tgt_t in task_dag_g.edges():
        dependencies_list.append({
            "source": src_t.id,
            "target": tgt_t.id
        })
    
    return {"tasks": tasks_list, "dependencies": dependencies_list}

def dynamic_task_dag_to_json(task_dag_g):
    return task_dag_to_json(task_dag_g)


def generate_task_dag_by_topology(num_initial_tasks: int = 30,
                                  max_unfold_iterations: int = 3,
                                  topology_type: str = "random",
                                  base_filename_prefix: str = "T0",
                                  output_visualization_dir=None,
                                  verbose: bool = True):
    return generate_dynamic_task_dag(
        num_initial_tasks=num_initial_tasks,
        max_unfold_iterations=max_unfold_iterations,
        base_filename_prefix=base_filename_prefix,
        output_visualization_dir=output_visualization_dir,
        dag_topology=topology_type,
        verbose=verbose,
    )

def main():
    parser = argparse.ArgumentParser(description='Generate Task DAGs')
    parser.add_argument('--num_tasks', type=int, default=15, help='Number of initial tasks for G_raw in dynamic DAG, or tasks in static DAG')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--dynamic', action='store_true', help='Generate dynamic DAG instead of static')
    parser.add_argument('--max_unfold', type=int, default=3, help='Max unfold iterations for SCCs in dynamic DAG')
    parser.add_argument('--topology', type=str, choices=['random', 'wide', 'deep'], default='random', help='DAG topology mode for dynamic generation')
    parser.add_argument('--prefix', type=str, default="dag", help="Prefix for output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dynamic:
        print(f"Generating dynamic DAG with {args.num_tasks} initial tasks, max unfold {args.max_unfold}...")
        final_dag = generate_dynamic_task_dag(
            num_initial_tasks=args.num_tasks, 
            max_unfold_iterations=args.max_unfold,
            base_filename_prefix=args.prefix.upper(),
            output_visualization_dir=args.output_dir, # Pass output_dir here
            dag_topology=args.topology,
        ) 
        
        with open(os.path.join(args.output_dir, f"{args.prefix}_dynamic_final.json"), "w") as f:
            json.dump(dynamic_task_dag_to_json(final_dag), f, indent=2)
        print(f"Dynamic DAG visualizations saved to '{args.output_dir}' and JSON saved with prefix '{args.prefix}'")
        # Also visualize the final DAG if task_generator.py is run directly
        _visualize_single_dag(final_dag, os.path.join(args.output_dir, f"{args.prefix}_final_DAG_from_script_main.png"))

    else: 
        print(f"Generating static DAG with {args.num_tasks} tasks...")
        Task.reset_counter() 
        static_dag = generate_task_dag(args.num_tasks)
        if static_dag.number_of_nodes() > 0:
            visualize_task_dag(static_dag, os.path.join(args.output_dir, f"{args.prefix}_static")) 
            with open(os.path.join(args.output_dir, f"{args.prefix}_static.json"), "w") as f:
                json.dump(task_dag_to_json(static_dag), f, indent=2)
            print(f"Static DAG visualization and JSON saved as '{args.prefix}_static.*' in '{args.output_dir}'")
        else:
            print("Static DAG generation resulted in an empty graph.")

if __name__ == "__main__":
    main()