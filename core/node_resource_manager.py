# node_resource_manager.py
"""
多节点任务调度系统 - 单节点资源管理器
负责管理单个节点的计算、内存和模型资源
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

@dataclass
class NodeResourceManager:
    """单节点资源管理器，负责管理节点的计算、内存和模型资源"""
    total_flops: int  # 总计算能力（每秒浮点运算次数）
    total_memory: int  # 总内存大小（MB）
    available_models: List[str]  # 可用的算法模型列表
    node_id: str = "unknown"  # 节点ID
    
    def __init__(self, total_flops: int, total_memory: int, available_models: List[str], node_id: str = "unknown"):
        self.total_flops = total_flops
        self.total_memory = total_memory
        self.available_models = available_models
        self.node_id = node_id
        self.base_total_flops = total_flops
        self.base_total_memory = total_memory
        self.used_flops = 0
        self.used_memory = 0
        self.peak_used_flops = 0  # 跟踪峰值计算资源使用情况
        self.peak_used_memory = 0  # 跟踪峰值内存使用情况
        
        # 资源使用记录 - 限制历史记录大小，防止内存泄漏
        self.resource_history = []  # 存储格式: [(start_time, end_time, flops, memory)]
        self.max_history_size = 1000  # 最大历史记录数量
        self.last_cleanup_time = time.time()  # 上次清理时间
        self.cleanup_interval = 10  # 清理间隔（秒）
        
        # 计算清空时间权重平均值的累计变量
        self.last_update_time = 0.0
        self.simulation_start_time = 0.0
        
        self.failed_tasks = {"memory_exceeded": 0, "model_unavailable": 0, "compute_exceeded": 0}

        # OPT: Cache for get_resource_status
        self._status_cache = None
        self._status_cache_dirty = True
        self._status_cache_time = -1.0

        # OPT: Dict mapping for O(1) release_resources lookup
        self._active_alloc_map = {}  # (flops, mem) -> [index, ...]

        # OPT: Incremental tracking for O(1) calculate_resource_usage
        # Instead of recalculating from history every time, maintain running sums.
        # On each allocate/release at time T:
        #   flush accumulated: _flops_time_sum += used_flops * (T - _last_event_time)
        #   then update used_flops/used_memory, set _last_event_time = T
        # calculate_resource_usage(end_time) just does one final flush: O(1)
        self._flops_time_sum = 0.0   # Σ(flops × duration) for completed intervals
        self._memory_time_sum = 0.0  # Σ(memory × duration) for completed intervals
        self._last_event_time = 0.0  # Time of last allocate/release event

    def exceeds_system_limits(self, compute_flops: int, memory_mb: int, model_name: str) -> Tuple[bool, str]:
        """检查任务资源需求是否超过系统总上限，返回(是否超限, 原因)"""
        # 检查模型是否可用
        if model_name not in self.available_models:
            reason = f"模型 {model_name} 不可用"
            self.failed_tasks["model_unavailable"] += 1
            return True, reason
            
        # 检查计算需求是否超出节点上限
        if compute_flops > self.total_flops:
            reason = f"计算需求超出节点上限: 需要 {compute_flops}，节点上限 {self.total_flops}"
            self.failed_tasks["compute_exceeded"] += 1
            return True, reason
            
        # 检查内存需求是否超出节点上限
        if memory_mb > self.total_memory:
            reason = f"内存需求超出节点上限: 需要 {memory_mb}MB，节点上限 {self.total_memory}MB"
            self.failed_tasks["memory_exceeded"] += 1
            return True, reason
            
        # 如果没有超过任何上限，返回 False
        return False, ""
    
    def has_available_resources(self, compute_flops: int, memory_mb: int) -> bool:
        """检查当前是否有足够的可用资源运行任务"""
        # 检查当前是否有足够的计算资源
        if self.used_flops + compute_flops > self.total_flops:
            return False
            
        # 检查当前是否有足够的内存资源
        if self.used_memory + memory_mb > self.total_memory:
            return False
            
        return True
        
    def can_run_task(self, compute_flops: int, memory_mb: int, model_name: str) -> bool:
        """检查是否有足够的资源运行任务"""
        # 首先检查是否超过系统上限
        exceeds, _ = self.exceeds_system_limits(compute_flops, memory_mb, model_name)
        if exceeds:
            return False
            
        # 然后检查当前资源是否足够
        return self.has_available_resources(compute_flops, memory_mb)
        
    def _flush_incremental(self, current_time: float):
        """OPT: Flush accumulated resource×time since last event into running sums."""
        if current_time > self._last_event_time:
            dt = current_time - self._last_event_time
            self._flops_time_sum += self.used_flops * dt
            self._memory_time_sum += self.used_memory * dt
            self._last_event_time = current_time

    def allocate_resources(self, compute_flops: int, memory_mb: int, model_name: str = None, current_time: float = 0.0) -> bool:
        """分配资源给任务"""
        if model_name is None or self.can_run_task(compute_flops, memory_mb, model_name):
            # 记录模拟开始时间，如果这是第一个任务
            if not self.resource_history and current_time > 0:
                self.simulation_start_time = current_time
                self.last_update_time = current_time
                self._last_event_time = current_time

            # OPT: Flush incremental before changing used_*
            self._flush_incremental(current_time)

            # 更新资源使用情况
            self.used_flops += compute_flops
            self.used_memory += memory_mb
            
            # 更新峰值使用情况
            self.peak_used_flops = max(self.peak_used_flops, self.used_flops)
            self.peak_used_memory = max(self.peak_used_memory, self.used_memory)
            
            # 记录资源分配，结束时间为None表示正在使用
            idx = len(self.resource_history)
            self.resource_history.append((current_time, None, compute_flops, memory_mb))
            # OPT: Track active allocation for O(1) release
            key = (compute_flops, memory_mb)
            if key not in self._active_alloc_map:
                self._active_alloc_map[key] = []
            self._active_alloc_map[key].append(idx)

            # OPT: Mark cache dirty
            self._status_cache_dirty = True

            # 清理过期的历史记录，防止内存泄漏
            self._cleanup_resource_history()
            
            # 更新最后更新时间
            self.last_update_time = current_time
            
            return True
        return False
        
    def release_resources(self, compute_flops: int, memory_mb: int, current_time: float = 0.0):
        """释放任务占用的资源"""
        # OPT: Flush incremental before changing used_*
        self._flush_incremental(current_time)

        # OPT: Use _active_alloc_map for O(1) lookup instead of linear scan
        key = (compute_flops, memory_mb)
        idx_list = self._active_alloc_map.get(key)
        if idx_list:
            idx = idx_list.pop()  # Remove last (most recent) matching allocation
            if not idx_list:
                del self._active_alloc_map[key]
            record = self.resource_history[idx]
            self.resource_history[idx] = (record[0], current_time, record[2], record[3])
        else:
            # Fallback: linear scan if map somehow missed
            for i in range(len(self.resource_history) - 1, -1, -1):
                record = self.resource_history[i]
                if record[1] is None and record[2] == compute_flops and record[3] == memory_mb:
                    self.resource_history[i] = (record[0], current_time, record[2], record[3])
                    break

        self.used_flops = max(0, self.used_flops - compute_flops)
        self.used_memory = max(0, self.used_memory - memory_mb)

        # OPT: Mark cache dirty
        self._status_cache_dirty = True
        
    def release_model(self, model_name: str):
        """释放模型资源，由于模型是共享资源，这里不需要特殊处理"""
        pass

    def apply_capacity_adjustment(self, flops_factor: float = 1.0, memory_factor: float = 1.0):
        flops_factor = max(0.05, flops_factor)
        memory_factor = max(0.05, memory_factor)
        self.total_flops = max(1, int(self.base_total_flops * flops_factor))
        self.total_memory = max(1, int(self.base_total_memory * memory_factor))
        self.used_flops = min(self.used_flops, self.total_flops)
        self.used_memory = min(self.used_memory, self.total_memory)

    def restore_capacity(self):
        self.total_flops = self.base_total_flops
        self.total_memory = self.base_total_memory
        self.used_flops = min(self.used_flops, self.total_flops)
        self.used_memory = min(self.used_memory, self.total_memory)

    def reset_simulation_time(self, start_time: float = 0.0):
        """重置模拟时间，在模拟开始前调用"""
        self.simulation_start_time = start_time
        self.last_update_time = start_time
    
    def calculate_resource_usage(self, end_time: float) -> Dict:
        """基于增量追踪计算平均资源使用率 - OPT: O(1) instead of O(R log R)
        
        Uses _flops_time_sum and _memory_time_sum maintained by allocate/release.
        Only needs a final flush from _last_event_time to end_time.
        
        Args:
            end_time: 模拟结束时间
        """
        if end_time <= self.simulation_start_time:
            return {'avg_flops_percent': 0, 'avg_memory_percent': 0, 'avg_flops': 0, 'avg_memory': 0}

        total_simulation_time = end_time - self.simulation_start_time
        if total_simulation_time <= 0:
            return {'avg_flops_percent': 0, 'avg_memory_percent': 0, 'avg_flops': 0, 'avg_memory': 0}

        # Final flush: account for time from last event to end_time
        # NOTE: we compute this without modifying _flops_time_sum/_last_event_time
        # because this is a read-only query (end_time may be intermediate)
        flops_sum = self._flops_time_sum
        memory_sum = self._memory_time_sum
        if end_time > self._last_event_time:
            dt = end_time - self._last_event_time
            flops_sum += self.used_flops * dt
            memory_sum += self.used_memory * dt
        
        avg_flops = flops_sum / total_simulation_time
        avg_memory = memory_sum / total_simulation_time
        avg_flops_percent = (avg_flops / self.total_flops) * 100 if self.total_flops > 0 else 0
        avg_memory_percent = (avg_memory / self.total_memory) * 100 if self.total_memory > 0 else 0
        
        return {
            'avg_flops_percent': avg_flops_percent,
            'avg_memory_percent': avg_memory_percent,
            'avg_flops': avg_flops,
            'avg_memory': avg_memory
        }
        
    def _cleanup_resource_history(self):
        """清理历史记录以防止内存泄漏 - OPT: only runs when over limit"""
        if len(self.resource_history) <= self.max_history_size:
            return

        # Simple approach: keep only active allocations + most recent completed
        active = [(i, r) for i, r in enumerate(self.resource_history) if r[1] is None]
        completed = [(i, r) for i, r in enumerate(self.resource_history) if r[1] is not None]

        # Keep all active, trim completed to fit budget
        budget = max(0, self.max_history_size - len(active))
        if len(completed) > budget:
            # Keep most recent completed records
            completed.sort(key=lambda x: x[1][1] or 0)  # Sort by end_time
            completed = completed[-budget:] if budget > 0 else []

        # Rebuild history preserving order
        kept_indices = set(i for i, _ in active) | set(i for i, _ in completed)
        self.resource_history = [self.resource_history[i] for i in sorted(kept_indices)]

        # Rebuild _active_alloc_map with new indices
        self._active_alloc_map.clear()
        for new_idx, (old_idx, r) in enumerate(active):
            key = (r[2], r[3])
            if key not in self._active_alloc_map:
                self._active_alloc_map[key] = []
            self._active_alloc_map[key].append(new_idx)
    
    def get_resource_status(self, current_time: float = None) -> Dict:
        """获取资源使用状态 - OPT: cached until state changes"""
        if current_time is None:
            current_time = self.last_update_time

        # OPT: Return cached result if not dirty and time matches
        if self._status_cache is not None and not self._status_cache_dirty and self._status_cache_time == current_time:
            return self._status_cache

        # 计算平均资源使用率 - 使用实际的当前时间，而不是固定的1000s
        avg_usage = self.calculate_resource_usage(current_time)
        
        result = {
            'node_id': self.node_id,
            'total_flops': self.total_flops,
            'used_flops': self.used_flops,  # 当前使用量
            'peak_used_flops': self.peak_used_flops,  # 峰值计算资源使用
            'avg_used_flops': avg_usage['avg_flops'],  # 平均计算资源使用
            'avg_flops_percent': avg_usage['avg_flops_percent'],  # 平均计算资源占用率
            'available_flops': self.total_flops - self.used_flops,
            'total_memory': self.total_memory,
            'used_memory': self.used_memory,  # 当前使用量
            'peak_used_memory': self.peak_used_memory,  # 峰值内存使用
            'avg_used_memory': avg_usage['avg_memory'],  # 平均内存使用
            'avg_memory_percent': avg_usage['avg_memory_percent'],  # 平均内存占用率
            'available_memory': self.total_memory - self.used_memory,
            'available_models': self.available_models,
            'failed_tasks': self.failed_tasks
        }

        # OPT: Cache result and mark clean
        self._status_cache = result
        self._status_cache_time = current_time
        self._status_cache_dirty = False

        return result
