# log_manager.py
import json
from typing import Dict, List

import logging
from typing import Dict, List
import json
import os

from project_paths import LOGS_DIR, ensure_results_layout

class LogManager:
    def __init__(self, clean_logs=True, verbose=True):
        ensure_results_layout()
        self.current_sim_time = 0.0  # 初始化仿真时间，使用浮点数
        self.log_file_path = str(LOGS_DIR / "system_status.log")
        self.task_lifecycle_path = str(LOGS_DIR / "task_lifecycle.log")
        self.console_log_path = str(LOGS_DIR / "console_output.log")
        self.task_events: Dict[str, List[Dict]] = {}  # 用于存储每个任务的事件记录
        self.verbose = verbose  # 控制是否输出详细信息到控制台
        
        # 初始化时清空日志文件（如果需要）
        if clean_logs:
            open(self.log_file_path, 'w').close()
            open(self.task_lifecycle_path, 'w').close()
            open(self.console_log_path, 'w').close()
        
        # 初始化控制台日志记录器
        self.setup_console_logger()
        
    def update_sim_time(self, sim_time):
        """更新当前仿真时间"""
        self.current_sim_time = round(float(sim_time), 2)
        
    def log_initial_state(self, nodes, node_resources):
        """Log the initial system state"""
        # Create initial system status
        status_info = {
            'queued_tasks': 0,
            'running_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'running_tasks_info': [],
            'failed_tasks_detail': [],
            'resource_failed_counts': {
                'memory_exceeded': 0,
                'model_unavailable': 0,
                'compute_exceeded': 0
            }
        }
        
        # Add resource status for each node
        all_resource_status = {}
        for node in nodes:
            node_id = node['id']
            resource_manager = node_resources[node_id]
            all_resource_status[node_id] = resource_manager.get_resource_status()
        
        # For simplicity, just use first node's resource status in the log
        if nodes:
            first_node = nodes[0]['id']
            status_info['resource_status'] = all_resource_status[first_node]
        
        # Write to log
        self.write_status(status_info)
    
    def log_system_state(self, simulation_time, nodes, node_resources, task_states, 
                         node_running_tasks, ready_tasks, completed_tasks, failed_tasks):
        """Log the current system state"""
        # Make a copy of ready_tasks to avoid modifying the heap
        ready_tasks_copy = ready_tasks.copy() if ready_tasks else []
        
        # Count tasks by status
        running_count = sum(1 for state in task_states.values() if state['status'] == 'running')
        queued_count = len(ready_tasks_copy)
        completed_count = len(completed_tasks)
        failed_count = len(failed_tasks)
        
        # Gather information about running tasks
        running_tasks_info = []
        for task_id, state in task_states.items():
            if state['status'] == 'running':
                running_tasks_info.append({
                    'task_id': task_id,
                    'model': state['required_model'],
                    'node': state['node_assigned']
                })
        
        # Gather information about failed tasks
        failed_tasks_detail = []
        for task_id in failed_tasks:
            if task_id in task_states and task_states[task_id]['failure_reason']:
                failed_tasks_detail.append({
                    'task_id': task_id,
                    'reason': task_states[task_id]['failure_reason']
                })
        
        # Count failed tasks by reason
        resource_failed_counts = {
            'memory_exceeded': 0,
            'model_unavailable': 0,
            'compute_exceeded': 0
        }
        
        for task_id in failed_tasks:
            if task_id in task_states:
                reason = task_states[task_id]['failure_reason']
                if reason and '内存需求超出' in reason:
                    resource_failed_counts['memory_exceeded'] += 1
                elif reason and '模型' in reason and '不可用' in reason:
                    resource_failed_counts['model_unavailable'] += 1
                elif reason and '计算需求超出' in reason:
                    resource_failed_counts['compute_exceeded'] += 1
        
        # Create status info object
        status_info = {
            'queued_tasks': queued_count,
            'running_tasks': running_count,
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'running_tasks_info': running_tasks_info,
            'failed_tasks_detail': failed_tasks_detail,
            'resource_failed_counts': resource_failed_counts
        }
        
        # Add resource status for all nodes
        all_resource_status = {}
        total_used_flops = 0
        total_used_memory = 0
        total_flops = 0
        total_memory = 0
        all_models = set()
        
        for node in nodes:
            node_id = node['id']
            resource_manager = node_resources[node_id]
            status = resource_manager.get_resource_status()
            all_resource_status[node_id] = status
            
            # Aggregate resources across all nodes for reporting
            total_used_flops += status['used_flops']
            total_used_memory += status['used_memory']
            total_flops += status['total_flops']
            total_memory += status['total_memory']
            all_models.update(status['available_models'])
        
        # Use aggregated resource stats for logging
        status_info['resource_status'] = {
            'used_flops': total_used_flops,
            'total_flops': total_flops,
            'used_memory': total_used_memory,
            'total_memory': total_memory,
            'available_models': list(all_models)
        }
        
        # Write status to log
        self.write_status(status_info)

    def log_system_status(self, status_info):
        """接收状态信息对象并记录系统状态"""
        # 直接将状态信息写入日志
        self.write_status(status_info)
        
    def write_status(self, status_info):
        """将状态信息写入日志文件，以追加模式记录每次状态更新"""
        # OPT: Skip file I/O in non-verbose mode to reduce simulation overhead
        if not self.verbose:
            return
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            import json
            # 写入仿真时间信息
            status_record = {
                "simulation_time": round(self.current_sim_time, 2),  # 统一使用仿真时间，保甲2位小数
                "task_stats": {
                    "queued": status_info['queued_tasks'],
                    "running": status_info['running_tasks'],
                    "completed": status_info['completed_tasks'],
                    "failed": status_info.get('failed_tasks', 0)
                },
                "running_tasks": [
                    {
                        "task_id": task['task_id'],
                        "model": task['model'],
                        "start_time": round(next((event['time'] for event in self.task_events.get(task['task_id'], []) if event['event_type'] == '开始执行'), self.current_sim_time), 2)
                    }
                    for task in status_info['running_tasks_info']
                ],
                # 添加失败任务的详细信息
                "failed_tasks": status_info.get('failed_tasks_detail', []),
                # 添加资源失败统计
                "resource_failed_counts": status_info.get('resource_failed_counts', {}),
                "resource_status": {
                    "compute": {
                        "used": status_info['resource_status']['used_flops'],
                        "total": status_info['resource_status']['total_flops'],
                        "unit": "FLOPS"
                    },
                    "memory": {
                        "used": status_info['resource_status']['used_memory'],
                        "total": status_info['resource_status']['total_memory'],
                        "unit": "MB"
                    },
                    "available_models": status_info['resource_status']['available_models']
                }
            }
            
            # 将状态记录以JSON格式写入文件，每条记录后添加换行符
            json.dump(status_record, f, ensure_ascii=False, indent=2)
            f.write('\n')

    def log_task_lifecycle(self, task_id, event_type, event_info):
        """记录任务生命周期事件
        
        Args:
            task_id: 任务ID
            event_type: 事件类型（创建/开始执行/完成执行/任务失败）
            event_info: 事件相关信息（字典格式）
        """
        # 初始化任务事件列表（如果不存在）
        if task_id not in self.task_events:
            self.task_events[task_id] = []
        
        # 记录所有类型的任务事件
        event = {
            'time': round(self.current_sim_time, 2),
            'event_type': event_type,
            'event_info': event_info
        }
        self.task_events[task_id].append(event)
        
        # 注意: 我们不再在每次事件记录时重写日志文件
        # 而是在finalize_task_lifecycle_log方法中一次性写入所有事件
    
    def summarize_task_events(self):
        """从任务事件中提取任务摘要信息"""
        task_summaries = {}
        
        # 按任务ID和时间排序
        sorted_tasks = sorted(self.task_events.items(), key=lambda x: x[0])
        
        for tid, events in sorted_tasks:
            # 按时间顺序排序事件
            sorted_events = sorted(events, key=lambda x: x['time'])
            
            # 初始化必要的字段
            creation_time = 0.0  # 默认创建时间为0
            start_time = None
            completion_time = -1
            failure_reason = None
            static_fields = {
                "required_flops": None,
                "required_memory": None,
                "required_model": None
            }
            
            # 处理所有事件
            for event in sorted_events:
                # 记录模型和资源需求信息，不管事件类型
                if 'event_info' in event and 'required_flops' in event['event_info']:
                    static_fields["required_flops"] = event['event_info'].get('required_flops')
                if 'event_info' in event and 'required_memory' in event['event_info']:
                    static_fields["required_memory"] = event['event_info'].get('required_memory')
                if 'event_info' in event and 'required_model' in event['event_info']:
                    static_fields["required_model"] = event['event_info'].get('required_model')
                
                # 任务创建事件
                if event['event_type'] == '任务创建':
                    creation_time = event['time']
                
                # 开始执行事件
                elif event['event_type'] == '开始执行':
                    start_time = event['time']
                    # 也可能包含资源需求信息
                    if 'event_info' in event and 'start_time' in event['event_info']:
                        start_time = event['event_info']['start_time']
                    
                # 完成执行事件
                elif event['event_type'] == '完成执行' or event['event_type'] == '任务完成':
                    completion_time = event['time']
                    
                # 任务失败事件
                elif event['event_type'] == '任务失败':
                    failure_reason = event['event_info'].get('reason')
                
                # 任务启动事件 - 可能包含任务信息
                elif event['event_type'] == '任务启动':
                    if 'event_info' in event and 'start_time' in event['event_info']:
                        start_time = event['event_info']['start_time']
            
            # 查找节点信息
            node_id = None
            for event in sorted_events:
                if 'event_info' in event and 'node' in event['event_info']:
                    node_id = event['event_info']['node']
                    break
            
            # 构建简化后的任务记录
            task_summary = {
                "task_id": tid,
                "required_flops": static_fields["required_flops"],
                "required_memory": static_fields["required_memory"],
                "required_model": static_fields["required_model"],
                "creation_time": round(creation_time, 2),
                "start_time": round(start_time, 2) if start_time is not None else None,
                "completion_time": round(completion_time, 2) if completion_time != -1 else None,
                "node": node_id,  # 添加节点ID
                "failure_reason": failure_reason
            }
            task_summaries[tid] = task_summary
        
        return task_summaries
    
    def finalize_task_lifecycle_log(self):
        """将所有任务生命周期事件生成统一的JSON格式并写入日志文件"""
        # 生成每个任务的摘要信息
        task_summaries = self.summarize_task_events()
        
        # 创建最终的输出格式
        output_data = {
            "simulation_time": round(self.current_sim_time, 2),
            "tasks": list(task_summaries.values())
        }
        
        # 写入日志文件
        with open(self.task_lifecycle_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def setup_console_logger(self):
        """设置控制台日志记录器，将所有print输出重定向到日志文件"""
        # 创建日志记录器
        self.console_logger = logging.getLogger('console')
        self.console_logger.setLevel(logging.INFO)
        
        # 防止日志消息传播到根记录器
        self.console_logger.propagate = False
        
        # 清除任何现有的处理程序
        for handler in self.console_logger.handlers[:]: 
            self.console_logger.removeHandler(handler)
        
        # 创建文件处理程序
        file_handler = logging.FileHandler(self.console_log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        # 只有在verbose模式下才添加控制台处理程序
        self.console_logger.addHandler(file_handler)
        
        if self.verbose:
            # 创建控制台处理程序，以便在控制台也能看到输出
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            # 添加控制台处理程序
            self.console_logger.addHandler(console_handler)
    
    def log_print(self, message):
        """记录一条打印消息到日志文件，同时在verbose模式下在控制台显示"""
        # 使用设置好的记录器记录消息
        if self.verbose or message.startswith("["): # 确保进度信息仍然显示
            self.console_logger.info(message)