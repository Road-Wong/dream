# topology_generator.py
import random
import networkx as nx
import json
import os

# 全局模型定义
def generate_model_versions(prefix, num_versions):
    return [f"{prefix}_v{i}" for i in range(1, num_versions + 1)]

# 注意这部分定义了模型的型号和版本，与task_generator.py中的保持一致
def generate_all_models(small_versions=2, medium_versions=2, large_versions=1, super_large_versions=1):
    # 减少模型的版本数量，使分配变得更加容易管理
    small_models = generate_model_versions("small", small_versions)
    medium_models = generate_model_versions("medium", medium_versions)
    large_models = generate_model_versions("large", large_versions)
    super_large_models = generate_model_versions("super_large", super_large_versions)
    return small_models + medium_models + large_models + super_large_models

class Node:
    id_counter = 0
    
    def __init__(self, node_type, compute_power, memory):
        Node.id_counter += 1
        prefix = 'D' if node_type == 'edge' else node_type[0].upper()
        self.id = f"{prefix}{Node.id_counter}"
        self.node_type = node_type
        self.compute_power = compute_power
        self.memory = memory
        self.models = []

    def __str__(self):
        return f"{self.id} ({self.node_type})"

def get_model_type(model_name):
    """从模型名称中提取模型类型"""
    if model_name.startswith("small_"):
        return "small"
    elif model_name.startswith("medium_"):
        return "medium"
    elif model_name.startswith("large_"):
        return "large"
    elif model_name.startswith("super_large_"):
        return "super_large"
    else:
        return None

def allocate_models_by_node_type(nodes):
    """根据节点类型分配模型
    
    分配规则：
    - Cloud Nodes: 可以分配任意类型的模型
    - Edge Nodes: 只能分配small、medium和large类型的模型
    - End Nodes: 只能分配small类型的模型
    
    改进：确保每种模型至少分配给一个节点，避免出现无法执行的任务
    """
    # 生成所有可用模型
    all_models = generate_all_models()
    all_models_flat = all_models
    
    # 按节点类型分组
    cloud_nodes = [node for node in nodes if node.node_type == "cloud"]
    edge_nodes = [node for node in nodes if node.node_type == "edge"]
    end_nodes = [node for node in nodes if node.node_type == "end"]
    
    # 1. 确保所有super_large模型至少分配给一个cloud节点
    super_large_models = [model for model in all_models if model.startswith("super_large_")]
    
    # 2. 确保所有large模型至少分配给一个cloud或edge节点
    large_models = [model for model in all_models if model.startswith("large_")]
    
    # 3. 确保所有medium模型至少分配给一个cloud或edge节点
    medium_models = [model for model in all_models if model.startswith("medium_")]
    
    # 4. 确保所有small模型至少分配给一个节点（包括end节点）
    small_models = [model for model in all_models if model.startswith("small_")]
    
    # 为每种类型的节点分配模型，确保覆盖所有模型类型
    if cloud_nodes:  # 如果有云节点
        # 确保每个super_large模型至少分配给一个cloud节点
        for i, model in enumerate(super_large_models):
            cloud_idx = i % len(cloud_nodes)  # 循环分配
            if model not in cloud_nodes[cloud_idx].models:
                cloud_nodes[cloud_idx].models.append(model)
    
        # 确保其他模型也有分配
        for node in cloud_nodes:
            num_models = random.randint(7, 10)
            # 添加随机模型，但避免重复
            potential_models = [m for m in all_models if m not in node.models]
            node.models.extend(random.sample(potential_models, 
                                min(num_models - len(node.models), 
                                    len(potential_models))))
    
    # 为edge节点分配模型
    if edge_nodes:  # 如果有边缘节点
        allowed_models = medium_models + large_models
        # 确保每个large和medium模型至少分配给一个edge节点
        for i, model in enumerate(allowed_models):
            edge_idx = i % len(edge_nodes)  # 循环分配
            if model not in edge_nodes[edge_idx].models:
                edge_nodes[edge_idx].models.append(model)
    
        # 为每个edge节点添加额外的模型
        for node in edge_nodes:
            allowed_models_for_edge = [model for model in all_models 
                                      if not model.startswith("super_large_") and model not in node.models]
            num_extra = random.randint(1, 3)
            node.models.extend(random.sample(allowed_models_for_edge, 
                                min(num_extra, len(allowed_models_for_edge))))
    
    # 为end节点分配模型
    if end_nodes:  # 如果有端节点
        # 确保每个small模型至少分配给一个end节点
        for i, model in enumerate(small_models):
            end_idx = i % len(end_nodes)  # 循环分配
            if model not in end_nodes[end_idx].models:
                end_nodes[end_idx].models.append(model)
    
    return nodes

def validate_model_allocation(nodes):
    """验证模型分配是否符合规则"""
    for node in nodes:
        if node.node_type == "cloud":
            # Cloud Nodes可以分配任意类型的模型，无需验证
            continue
        elif node.node_type == "edge":
            # Edge Nodes只能分配small、medium和large类型的模型
            for model in node.models:
                model_type = get_model_type(model)
                if model_type == "super_large":
                    return False, f"Edge Node {node.id} 不应该分配 {model} 模型"
        elif node.node_type == "end":
            # End Nodes只能分配small类型的模型
            for model in node.models:
                model_type = get_model_type(model)
                if model_type != "small":
                    return False, f"End Node {node.id} 不应该分配 {model} 模型"
    
    return True, "模型分配符合规则"

def generate_resource_structure(num_cloud_nodes, num_edge_nodes, num_end_nodes):
    nodes = []
    
    # 创建云节点 - 确保能运行任何模型类型，包括super_large
    for _ in range(num_cloud_nodes):
        # 确保云节点有足够的算力处理super_large模型 (10000-300000)
        compute_power = random.randint(300000, 500000)  # 300,000-500,000 MFLOPS
        memory = random.randint(256, 1024)  # 256-1024 GB
        nodes.append(Node("cloud", compute_power, memory))
    
    # 创建边缘节点 - 确保能运行large模型 (3000-9000)
    for _ in range(num_edge_nodes):
        compute_power = random.randint(9000, 50000)  # 9,000-50,000 MFLOPS
        memory = random.randint(64, 256)  # 64-256 GB
        nodes.append(Node("edge", compute_power, memory))
    
    # 创建端节点 - 确保能运行small模型 (100-800)
    for _ in range(num_end_nodes):
        compute_power = random.randint(800, 5000)  # 800-5,000 MFLOPS
        memory = random.randint(8, 64)  # 8-64 GB
        nodes.append(Node("end", compute_power, memory))
    
    # 为节点分配模型
    nodes = allocate_models_by_node_type(nodes)
    
    # 验证模型分配结果
    valid, reason = validate_model_allocation(nodes)
    if not valid:
        print(f"模型分配验证失败: {reason}")
    
    return nodes

def create_network_topology(nodes):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    
    cloud_nodes = [node for node in nodes if node.node_type == "cloud"]
    edge_nodes = [node for node in nodes if node.node_type == "edge"]
    end_nodes = [node for node in nodes if node.node_type == "end"]

    # 连接云节点和边缘节点
    for cloud_node in cloud_nodes:
        for edge_node in edge_nodes:
            if random.random() < 0.7:  # 70% 的概率连接
                latency = random.uniform(10, 50)  # 10-50 ms
                bandwidth = random.randint(5000, 10000)  # 5000-10000 Mbps
                G.add_edge(cloud_node, edge_node, latency=latency, bandwidth=bandwidth)

    # 确保所有云节点和边缘节点都在同一个连通分量中
    if not nx.is_connected(G.subgraph(cloud_nodes + edge_nodes)):
        for i, node in enumerate(cloud_nodes + edge_nodes):
            if i > 0:
                prev_node = (cloud_nodes + edge_nodes)[i-1]
                if not nx.has_path(G, node, prev_node):
                    latency = random.uniform(10, 50)
                    bandwidth = random.randint(5000, 10000)  # 5000-10000 Mbps
                    G.add_edge(node, prev_node, latency=latency, bandwidth=bandwidth)

    # 将每个端节点连接到一个随机的边缘节点
    for end_node in end_nodes:
        edge_node = random.choice(edge_nodes)
        latency = random.uniform(1, 10)  # 1-10 ms
        bandwidth = random.randint(100, 1000)  # 100-1000 Mbps
        G.add_edge(end_node, edge_node, latency=latency, bandwidth=bandwidth)

    return G

def visualize_network_topology(G, filename):
    # 如果filename为None，则不生成可视化
    if filename is None:
        return
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
        
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)
    colors = ['red' if node.node_type == 'cloud' else 'green' if node.node_type == 'edge' else 'blue' for node in G.nodes()]
    
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=700, ax=ax)
    
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # 创建包含延迟和带宽的边标签
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = f"{data['latency']:.2f} ms\n{data['bandwidth']} Mbps"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Resource Network Topology")
    plt.savefig(filename)
    plt.close(fig)

def network_to_json(network):
    """将资源网络转换为JSON格式"""
    nodes = []
    for node in network.nodes():
        nodes.append({
            "id": node.id,
            "type": node.node_type,
            "compute_power": node.compute_power,
            "memory": node.memory,
            "models": node.models
        })
    
    edges = []
    for edge in network.edges(data=True):
        edges.append({
            "source": edge[0].id,
            "target": edge[1].id,
            "latency": edge[2]['latency'],
            "bandwidth": edge[2]['bandwidth']
        })
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def generate_network_ascii(network):
    """生成网络拓扑的ASCII文本表示"""
    ascii_graph = ""
    for node in network.nodes():
        neighbors = list(network.neighbors(node))
        if neighbors:
            ascii_graph += f"{node.id} -> {', '.join(neighbor.id for neighbor in neighbors)}\n"
        else:
            ascii_graph += f"{node.id}\n"
    return ascii_graph

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成网络拓扑结构')
    parser.add_argument('--cloud', type=int, default=3, help='云节点数量')
    parser.add_argument('--edge', type=int, default=5, help='边缘节点数量')
    parser.add_argument('--end', type=int, default=10, help='端节点数量')
    parser.add_argument('--output', type=str, default='topology_output', help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 生成系统模型
    nodes = generate_resource_structure(args.cloud, args.edge, args.end)
    network = create_network_topology(nodes)
    visualize_network_topology(network, f"{args.output}/network_topology.png")
    
    # 保存网络为JSON文件
    with open(f"{args.output}/network_topology.json", "w") as f:
        json.dump(network_to_json(network), f, indent=2)
    
    # 保存网络为ASCII文本文件
    with open(f"{args.output}/network_topology_ascii.txt", "w") as f:
        f.write("=== System Configuration ===\n")
        f.write(f"Cloud Nodes: {args.cloud}\n")
        f.write(f"Edge Nodes: {args.edge}\n")
        f.write(f"End Nodes: {args.end}\n\n")
        f.write("=== Model Allocation Rules ===\n")
        f.write("Cloud Nodes: 可以分配任意类型的模型\n")
        f.write("Edge Nodes: 只能分配small、medium和large类型的模型\n")
        f.write("End Nodes: 只能分配small类型的模型\n\n")
        f.write("=== Network Topology ===\n")
        f.write(generate_network_ascii(network))
    
    print(f"网络拓扑已生成并保存到 {args.output} 目录")

if __name__ == "__main__":
    main()
