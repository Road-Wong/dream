# network_sim.py
"""
网络延迟仿真模块
负责计算不同节点间的数据传输延迟，构建网络路由表，及计算端到端传输时延
"""
import heapq
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import simpy
import networkx as nx
from core.log_manager import LogManager

class NetworkSimulator:
    """网络延迟仿真器，计算节点间数据传输延迟"""
    
    def __init__(self, env: simpy.Environment, network: Dict, log_manager: LogManager):
        """
        初始化网络延迟仿真器
        
        参数:
            env: 模拟环境
            network: 网络拓扑定义
            log_manager: 日志管理器
        """
        self.env = env
        self.log_manager = log_manager
        self.nodes = network['nodes']
        self.edges = network['edges']
        self.node_map = {node['id']: node for node in self.nodes}

        self.network_profile = dict(network.get('network_profile', {}) or {})
        self.network_mode = str(self.network_profile.get('mode', 'static')).lower()
        if self.network_mode not in {'static', 'wired', 'wireless', 'dynamic'}:
            self.network_mode = 'static'

        def _safe_profile_float(key: str, default: float) -> float:
            try:
                return float(self.network_profile.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        self.contention_alpha = max(0.0, _safe_profile_float('contention_alpha', 0.0))
        self.wireless_contention_alpha = max(0.0, _safe_profile_float('wireless_contention_alpha', 0.0))
        self.time_variation_amplitude = max(0.0, _safe_profile_float('time_variation_amplitude', 0.0))
        self.time_variation_period = max(1e-6, _safe_profile_float('time_variation_period', 60.0))
        self.min_bandwidth_factor = max(0.05, _safe_profile_float('min_bandwidth_factor', 0.2))

        self.active_transfers_by_domain = defaultdict(int)
        self.active_transfer_total = 0

        # 构建图形结构用于计算最短路径
        self.graph = nx.DiGraph()
        
        # 添加所有节点
        for node in self.nodes:
            self.graph.add_node(node['id'])
        
        # 添加所有边及其权重（使用延迟作为权重）
        for edge in self.edges:
            # 添加边和延迟权重
            self.graph.add_edge(
                edge['source'], 
                edge['target'], 
                latency=edge['latency'],
                bandwidth=edge['bandwidth'],
                contention_domain=edge.get('contention_domain')
            )
            # 注意：假设网络是双向的，添加反向边
            self.graph.add_edge(
                edge['target'], 
                edge['source'], 
                latency=edge['latency'],
                bandwidth=edge['bandwidth'],
                contention_domain=edge.get('contention_domain')
            )
        
        # 预计算所有节点对之间的最短路径和总延迟
        self._build_routing_table()
        
    def _build_routing_table(self):
        """构建路由表，计算所有节点对之间的最短路径和延迟"""
        self.shortest_paths = {}
        self.path_latencies = {}
        self.path_bandwidths = {}
        self.path_domains = {}
        
        # 为所有节点对计算最短路径
        for source in self.graph.nodes():
            self.shortest_paths[source] = {}
            self.path_latencies[source] = {}
            self.path_bandwidths[source] = {}
            self.path_domains[source] = {}
            
            # 使用Dijkstra算法计算从source到所有其他节点的最短路径
            # 权重为链路的延迟
            paths = nx.single_source_dijkstra(self.graph, source, weight='latency')
            distances, paths_dict = paths
            
            for target in self.graph.nodes():
                if source != target:
                    if target in paths_dict:
                        # 存储最短路径
                        self.shortest_paths[source][target] = paths_dict[target]
                        
                        # 计算路径的总延迟
                        total_latency = distances[target]
                        self.path_latencies[source][target] = total_latency
                        
                        # 计算路径的最小带宽（瓶颈）
                        path = paths_dict[target]
                        min_bandwidth = float('inf')
                        domains = []
                        for i in range(len(path) - 1):
                            edge_data = self.graph[path[i]][path[i+1]]
                            edge_bandwidth = edge_data['bandwidth']
                            min_bandwidth = min(min_bandwidth, edge_bandwidth)
                            domain = edge_data.get('contention_domain')
                            if domain:
                                domains.append(str(domain))
                        self.path_bandwidths[source][target] = min_bandwidth
                        self.path_domains[source][target] = tuple(sorted(set(domains)))
                    else:
                        # 如果没有路径，设置为None
                        self.shortest_paths[source][target] = None
                        self.path_latencies[source][target] = float('inf')
                        self.path_bandwidths[source][target] = 0
                        self.path_domains[source][target] = tuple()

    def _get_path_domains(self, source_node: str, target_node: str) -> Tuple[str, ...]:
        domains = self.path_domains.get(source_node, {}).get(target_node, tuple())
        if domains:
            return domains
        if self.network_mode == 'wireless':
            return ('wireless_shared',)
        if self.network_mode == 'dynamic':
            return ('dynamic_shared',)
        return tuple()

    def _compute_time_variation_multiplier(self, source_node: str, target_node: str) -> float:
        if self.network_mode != 'dynamic':
            return 1.0

        amp = min(0.95, self.time_variation_amplitude)
        if amp <= 1e-12:
            return 1.0

        now = float(getattr(self.env, 'now', 0.0))
        pair_seed = abs(hash(f"{source_node}->{target_node}")) % 360
        phase = (pair_seed / 360.0) * 2.0 * math.pi
        angle = (2.0 * math.pi * now / self.time_variation_period) + phase
        return max(0.05, 1.0 + amp * math.sin(angle))

    def _compute_contention_multiplier(self, source_node: str, target_node: str) -> float:
        domains = self._get_path_domains(source_node, target_node)

        if not domains:
            if self.network_mode in {'wireless', 'dynamic'}:
                domain_load = self.active_transfer_total
            else:
                domain_load = 0
        else:
            domain_load = max(self.active_transfers_by_domain.get(domain, 0) for domain in domains)

        if domain_load <= 0:
            return 1.0

        alpha = self.contention_alpha
        if self.network_mode == 'wireless':
            alpha += self.wireless_contention_alpha
        elif self.network_mode == 'dynamic':
            alpha += 0.5 * self.wireless_contention_alpha

        if alpha <= 1e-12:
            return 1.0

        return 1.0 + alpha * float(domain_load)

    def _register_transfer_start(self, source_node: str, target_node: str) -> Tuple[str, ...]:
        domains = self._get_path_domains(source_node, target_node)
        self.active_transfer_total += 1
        for domain in domains:
            self.active_transfers_by_domain[domain] += 1
        return domains

    def _register_transfer_end(self, domains: Tuple[str, ...]):
        self.active_transfer_total = max(0, self.active_transfer_total - 1)
        for domain in domains:
            current = self.active_transfers_by_domain.get(domain, 0)
            if current <= 1:
                self.active_transfers_by_domain.pop(domain, None)
            else:
                self.active_transfers_by_domain[domain] = current - 1
    
    def get_transfer_delay(self, source_node: str, target_node: str, data_size_mb: float) -> float:
        """
        计算从源节点到目标节点传输特定大小数据所需的延迟
        
        参数:
            source_node: 源节点ID
            target_node: 目标节点ID
            data_size_mb: 数据大小（MB）
            
        返回:
            传输延迟（时间单位）
        """
        # 如果源节点和目标节点相同，不需要传输
        if source_node == target_node:
            return 0.0
        
        # 获取路径延迟和带宽
        if source_node not in self.path_latencies or target_node not in self.path_latencies[source_node]:
            if self.log_manager.verbose:
                self.log_manager.log_print(f"警告: 节点 {source_node} 到 {target_node} 之间没有路径")
            return float('inf')
        
        base_latency = self.path_latencies[source_node][target_node]  # 单位: ms
        bandwidth = self.path_bandwidths[source_node][target_node]    # 单位: MB/s
        
        if bandwidth == 0:
            if self.log_manager.verbose:
                self.log_manager.log_print(f"警告: 节点 {source_node} 到 {target_node} 之间的带宽为0")
            return float('inf')

        safe_data_size_mb = max(0.0, float(data_size_mb))

        time_variation_multiplier = self._compute_time_variation_multiplier(source_node, target_node)
        contention_multiplier = self._compute_contention_multiplier(source_node, target_node)

        latency_seconds = (base_latency / 1000.0) * time_variation_multiplier * contention_multiplier
        effective_bandwidth = bandwidth / max(self.min_bandwidth_factor, time_variation_multiplier * contention_multiplier)
        if effective_bandwidth <= 0:
            return float('inf')

        transmission_time = safe_data_size_mb / effective_bandwidth  # 单位: 秒
        total_delay = latency_seconds + transmission_time  # 单位: 秒
        
        return total_delay
    
    def transfer_data(self, source_node: str, target_node: str, data_size_mb: float, task_id: str) -> simpy.events.Event:
        """
        模拟数据传输过程，返回一个完成事件
        
        参数:
            source_node: 源节点ID
            target_node: 目标节点ID
            data_size_mb: 数据大小（MB）
            task_id: 相关任务ID，用于日志记录
            
        返回:
            完成传输的simpy事件
        """
        # 计算传输延迟
        delay = self.get_transfer_delay(source_node, target_node, data_size_mb)
        
        if delay == float('inf'):
            if self.log_manager.verbose:
                self.log_manager.log_print(f"错误: 无法传输任务 {task_id} 的数据从 {source_node} 到 {target_node}")
            # 返回一个已触发的事件作为失败标志
            return self.env.timeout(0)

        transfer_domains = self._register_transfer_start(source_node, target_node)
            
        # 记录传输开始
        if self.log_manager.verbose:
            domain_text = ",".join(transfer_domains) if transfer_domains else "none"
            self.log_manager.log_print(
                f"[{self.env.now:.2f}] 开始传输任务 {task_id} 的数据 ({data_size_mb:.2f} MB) 从 {source_node} 到 {target_node}, "
                f"预计延迟: {delay:.2f}s, mode={self.network_mode}, domains={domain_text}, active={self.active_transfer_total}"
            )
        
        # 创建传输事件
        transfer_event = self.env.timeout(delay)
        
        # 注册回调以记录传输完成
        def transfer_done(event):
            self._register_transfer_end(transfer_domains)
            if self.log_manager.verbose:
                self.log_manager.log_print(
                    f"[{self.env.now:.2f}] 完成传输任务 {task_id} 的数据从 {source_node} 到 {target_node}, "
                    f"实际延迟: {delay:.2f}s, active={self.active_transfer_total}"
                )
        
        transfer_event.callbacks.append(transfer_done)
        return transfer_event
        
    def print_routing_table(self):
        """打印路由表信息，用于调试"""
        if self.log_manager.verbose:
            self.log_manager.log_print("网络路由表:")
            for source in sorted(self.shortest_paths.keys()):
                for target in sorted(self.shortest_paths[source].keys()):
                    path = self.shortest_paths[source][target]
                    latency = self.path_latencies[source][target]
                    bandwidth = self.path_bandwidths[source][target]
                    
                    if path:
                        path_str = " -> ".join(path)
                        self.log_manager.log_print(f"  {source} -> {target}: 路径={path_str}, 延迟={latency:.2f}ms, 带宽={bandwidth:.2f}MB/s")
