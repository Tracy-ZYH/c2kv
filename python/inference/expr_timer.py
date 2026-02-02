import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class PhaseTimer:
    """单个阶段的计时上下文管理器"""
    
    def __init__(self, phase_name: str, recorder: 'DataRecorder', device_id: int = 0):
        self.phase_name = phase_name
        self.recorder = recorder
        self.device_id = device_id
        self.start_time = None
        self.enable = recorder.enable
    
    def __enter__(self):
        if not self.enable:
            return self
        if CUDA_AVAILABLE:
            torch.cuda.synchronize(device=self.device_id)
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return self
        if CUDA_AVAILABLE:
            torch.cuda.synchronize(device=self.device_id)
        elapsed = time.perf_counter() - self.start_time
        self.recorder.record_phase(self.phase_name, elapsed)
        return False


class DataRecorder:
    """单条数据的计时记录器"""
    
    def __init__(self, data_id: str, enable: bool = True, device_id: int = 0):
        self.data_id = data_id
        self.enable = enable
        self.device_id = device_id
        self.phases: Dict[str, float] = {}
        self.phase_order: List[str] = []  # 保持阶段顺序
    
    def record_phase(self, phase_name: str, elapsed: float):
        """记录单个阶段的耗时"""
        if not self.enable:
            return
        
        if phase_name not in self.phases:
            self.phase_order.append(phase_name)
            self.phases[phase_name] = 0.0
        
        self.phases[phase_name] += elapsed
    
    @contextmanager
    def record(self, phase_name: str):
        """上下文管理器：自动计时某个阶段"""
        timer = PhaseTimer(phase_name, self, self.device_id)
        with timer:
            yield timer
    
    def get_total_time(self) -> float:
        """获取总耗时"""
        return sum(self.phases.values())
    
    def get_phase_time(self, phase_name: str) -> float:
        """获取某个阶段的耗时"""
        return self.phases.get(phase_name, 0.0)
    
    def summary(self) -> Dict[str, float]:
        """返回此条数据的耗时总结（秒，4位小数）"""
        result = {}
        for phase in self.phase_order:
            result[phase] = round(self.phases[phase], 4)
        result['total'] = round(self.get_total_time(), 4)
        return result
    
    def __repr__(self):
        summary = self.summary()
        return f"DataRecord(id={self.data_id}, {summary})"


class ExprTimer:
    """实验级别的计时管理器"""
    
    def __init__(self, expr_name: str = "experiment", enable: bool = True, device_id: int = 0):
        self.expr_name = expr_name
        self.enable = enable
        self.device_id = device_id
        self.records: Dict[str, DataRecorder] = {}
        self.record_order: List[str] = []  # 保持记录顺序
        self.current_record: Optional[DataRecorder] = None
    
    def record(self, data_id: str) -> DataRecorder:
        """获取或创建一个新的数据记录器"""
        if data_id not in self.records:
            self.records[data_id] = DataRecorder(data_id, self.enable, self.device_id)
            self.record_order.append(data_id)
        
        self.current_record = self.records[data_id]
        return self.current_record
    
    def get_last_record(self) -> Optional[Dict[str, float]]:
        """获取上一条数据的计时总结"""
        if self.current_record is None:
            return None
        return self.current_record.summary()
    
    def get_all_records(self) -> Dict[str, Dict[str, float]]:
        """获取所有数据的详细计时"""
        return {data_id: self.records[data_id].summary() 
                for data_id in self.record_order}
    
    def statistics(self) -> Dict[str, Dict[str, float]]:
        """
        统计所有数据的平均耗时（按阶段和总计）
        
        Returns:
            {
                'phase_1': {'mean': 0.1234, 'std': 0.0045,},
                'phase_2': {...},
                'total': {...}
            }
        """
        if not self.records:
            return {}
        
        # 收集所有阶段
        all_phases = set()
        for record in self.records.values():
            all_phases.update(record.phases.keys())
        all_phases = sorted(all_phases)
        
        statistics = {}
        
        # 计算每个阶段的统计
        for phase in all_phases:
            times = [self.records[data_id].get_phase_time(phase) 
                     for data_id in self.record_order]
            times = np.array(times)
            
            statistics[phase] = {
                'mean': round(float(np.mean(times)), 4),
                'std': round(float(np.std(times)), 4),
            }
        
        # 计算总耗时的统计
        total_times = np.array([self.records[data_id].get_total_time() 
                                for data_id in self.record_order])
        statistics['total'] = {
            'mean': round(float(np.mean(total_times)), 4),
            'std': round(float(np.std(total_times)), 4),
        }
        
        return statistics
