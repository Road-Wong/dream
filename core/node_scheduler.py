# node_scheduler.py
import simpy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from core.log_manager import LogManager
from core.node_resource_manager import NodeResourceManager

@dataclass
class NodeTask:
    """单节点任务对象"""
    task_id: str
    compute_flops: int  # 所需浮点运算次数
    memory_mb: int     # 所需内存大小
    model_name: str    # 算法模型名称
    duration: float    # 执行时间
    status: str = 'pending'  # 任务状态：pending, running, completed, failed
    start_time: float = 0.0  # 任务实际开始时间，用于预测结束时刻

class NodeScheduler:
    """单节点调度器，负责管理单个节点上的任务队列和任务执行"""
    
    def __init__(self, env: simpy.Environment, resource_manager: NodeResourceManager, log_manager: LogManager, node_id: str = "unknown"):
        self.env = env
        self.resource_manager = resource_manager
        self.log_manager = log_manager
        self.node_id = node_id
        self.task_queue = []
        self.running_tasks: List[NodeTask] = []  # 支持多任务并行执行
        self.completed_tasks = []
        self.failed_tasks = []  # 存储因资源超限而失败的任务

    def schedule_task(self, task_data: Dict, completion_callback: Callable[[str, str], None] = None) -> bool:
        """安排任务到节点上执行，返回是否成功安排"""
        # 创建任务对象
        task = NodeTask(
            task_id=task_data['task_id'],
            compute_flops=task_data['compute_flops'],
            memory_mb=task_data['memory_mb'],
            model_name=task_data['model_name'],
            duration=task_data['duration']
        )
        # 只在verbose模式下输出调试信息
        if self.log_manager.verbose:
            self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} received with duration {task.duration:.2f}")
        
        # 检查任务资源需求是否超过系统上限
        exceeds_limits, failure_reason = self.resource_manager.exceeds_system_limits(
            task.compute_flops, 
            task.memory_mb, 
            task.model_name
        )
        
        # 如果资源需求超过系统上限，任务失败
        if exceeds_limits:
            task.status = 'failed'
            self.failed_tasks.append(task)
            
            # 记录任务失败事件
            self.log_manager.update_sim_time(self.env.now)
            self.log_manager.log_task_lifecycle(
                task.task_id,
                '任务失败',
                {
                    'status': '失败',
                    'reason': failure_reason
                }
            )
            
            # 任务失败日志，只在verbose模式下输出详细信息
            if self.log_manager.verbose:
                self.log_manager.log_print(f"[{self.env.now:.2f}] 任务 {task.task_id} (on NodeScheduler {self.node_id}) 因资源超限而失败: {failure_reason}")
            return False
        
        # 启动任务执行进程
        self.env.process(self._execute_task(task, completion_callback))
        return True

    def _execute_task(self, task: NodeTask, completion_callback: Optional[Callable[[str, str], None]] = None):
        """执行单个任务的仿真过程"""
        # 等待直到有足够的资源可用
        while not self.resource_manager.has_available_resources(task.compute_flops, task.memory_mb):
            # self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} waiting for resources at {self.env.now:.2f} (needs {task.compute_flops}F, {task.memory_mb}MB)")
            yield self.env.timeout(0.1) # Check every 0.1 time units
            
        # 分配资源
        # self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} attempting to allocate resources at {self.env.now:.2f}")
        allocated = self.resource_manager.allocate_resources(task.compute_flops, task.memory_mb, task.model_name, self.env.now)
        if not allocated: # Should be rare if has_available_resources passed, but as a safeguard
            # 关键错误日志，即使在非verbose模式下也应输出
            if self.log_manager.verbose:
                self.log_manager.log_print(f"CRITICAL ERROR NS {self.node_id}: Task {task.task_id} FAILED to allocate resources at {self.env.now:.2f} despite passing check.")
            # Handle as a failure if allocate_resources itself can fail and returns False
            task.status = 'failed'
            self.failed_tasks.append(task)
            self.log_manager.update_sim_time(self.env.now)
            self.log_manager.log_task_lifecycle(task.task_id, '任务失败', {'status': '失败', 'reason': 'Resource allocation failed internally', 'node': self.node_id})
            if completion_callback: # Notify ClusterManager of failure if possible, though CM might not expect this path
                 # This path needs careful thought; _fail_task in CM is usually called by CM itself.
                 # For now, just log and the task won't complete successfully.
                 pass
            return # Stop execution of this task

        # 更新日志管理器的仿真时间
        self.log_manager.update_sim_time(self.env.now)
        # 记录资源分配事件 - This might be redundant if CM logs it. For debugging, can keep.
        self.log_manager.log_task_lifecycle(
            task.task_id,
            '资源分配_NS', # Distinguish from CM log if any
            {
                'allocated_flops': task.compute_flops,
                'allocated_memory': task.memory_mb,
                'node': self.node_id,
                'time': self.env.now
            }
        )
        
        # 更新任务状态
        task.start_time = self.env.now
        task.status = 'running'
        self.running_tasks.append(task)
        
        # 更新日志管理器的仿真时间 - CM's '开始执行' log is more authoritative for task state
        # self.log_manager.update_sim_time(self.env.now)
        # self.log_manager.log_task_lifecycle(
        #     task.task_id,
        #     '开始执行_NS', # Distinguish
        #     {'status': '正在执行', 'node': self.node_id, 'time': self.env.now}
        # )
        
        # 模拟任务执行时间
        # 只在verbose模式下输出调试信息
        if self.log_manager.verbose:
            self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} SIMULATING for duration {task.duration:.2f} at time {self.env.now:.2f}")
        yield self.env.timeout(task.duration)
        
        # 完成任务并释放资源
        # self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} completed simulation at {self.env.now:.2f}, calling _complete_task.")
        self._complete_task(task)
        
        # 执行回调函数（如果有）
        if completion_callback:
            # self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} invoking completion_callback at {self.env.now:.2f}.")
            completion_callback(task.task_id, self.node_id)

    def _complete_task(self, task: NodeTask):
        """完成任务并释放资源"""
        # 更新任务状态
        task.status = 'completed'
        if task not in self.completed_tasks : self.completed_tasks.append(task) # Avoid duplicates if called multiple times
        
        # 更新日志管理器的仿真时间 - CM's '完成执行' log is more authoritative
        # self.log_manager.update_sim_time(self.env.now)
        # self.log_manager.log_task_lifecycle(
        #     task.task_id,
        #     '完成执行_NS', # Distinguish
        #     {'status': '执行完成', 'node': self.node_id, 'time': self.env.now }
        # )
        
        # 释放资源
        # self.log_manager.log_print(f"DEBUG NS {self.node_id}: Task {task.task_id} releasing resources at {self.env.now:.2f}.")
        self.resource_manager.release_resources(task.compute_flops, task.memory_mb, self.env.now)
        # 释放模型资源 - this does nothing in current NodeResourceManager
        self.resource_manager.release_model(task.model_name)
        
        # 更新日志管理器的仿真时间
        self.log_manager.update_sim_time(self.env.now)
        # 记录资源释放事件
        self.log_manager.log_task_lifecycle(
            task.task_id,
            '资源释放_NS', # Distinguish
            {'status': '资源已释放', 'node': self.node_id, 'time': self.env.now}
        )
        
        # 从正在运行的任务列表中移除
        if task in self.running_tasks:
            self.running_tasks.remove(task)

    def get_queue_status(self) -> Dict:
        """获取当前队列状态"""
        return {
            'current_time': self.env.now,
            'queued_tasks': len(self.task_queue), # task_queue is not actively used for queueing in current logic
            'running_tasks': len(self.running_tasks),
            'running_tasks_info': [{'task_id': task.task_id, 'model': task.model_name} for task in self.running_tasks],
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks), # These are NS-level failed tasks (resource exceeding limits)
            'resource_status': self.resource_manager.get_resource_status(current_time=self.env.now)
        }

    def predict_wait_time_for(self, compute_flops: int, memory_mb: int, now: float = None) -> float:
        """预测满足给定资源需求所需的等待时间（基于当前running_tasks的结束时刻）。"""
        now = self.env.now if now is None else now
        rm = self.resource_manager
        # 如果当前即刻可满足资源，则无需等待
        if rm.used_flops + compute_flops <= rm.total_flops and rm.used_memory + memory_mb <= rm.total_memory:
            return 0.0
        # 构建完成事件序列：任务完成后释放的资源
        events = []  # (finish_time, freed_flops, freed_mem)
        for t in self.running_tasks:
            ft = (t.start_time or now) + t.duration
            events.append((ft, t.compute_flops, t.memory_mb))
        if not events:
            # 没有运行中任务但仍无法满足，返回0（应该由系统上限检查阻挡这种情况）
            return 0.0
        events.sort(key=lambda x: x[0])
        avail_flops = rm.total_flops - rm.used_flops
        avail_mem = rm.total_memory - rm.used_memory
        for ft, f, m in events:
            avail_flops += f
            avail_mem += m
            if avail_flops >= compute_flops and avail_mem >= memory_mb:
                return max(0.0, ft - now)
        # 理论上在最后一个事件后应当足够
        return max(0.0, events[-1][0] - now)