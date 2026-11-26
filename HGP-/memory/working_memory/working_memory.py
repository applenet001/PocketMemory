"""
WorkingMemory 主控类
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

from .state_trajectory import StateTrajectoryGraph, StateNode, StateTransition, WORKING_MEMORY_CAPACITY


@dataclass
class WorkingMemoryCheckpoint:
    checkpoint_id: str                          
    timestamp: str                              
    current_state_id: str                    
    task_name: str                            
    total_states: int                          
    states_snapshot: Dict[str, StateNode]      
    context: Dict[str, Any]                     


class WorkingMemory:
    def __init__(
        self,
        task_name: str,
        storage_dir: Optional[Path] = None,
        capacity: int = WORKING_MEMORY_CAPACITY
    ):
        self.task_name = task_name
        
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "graph" / "working_memory"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        storage_path = self.storage_dir / "state_trajectory.json"
        self.trajectory = StateTrajectoryGraph(storage_path, event_id=task_name, capacity=capacity)

        self.context: Dict[str, Any] = {}
        
        self.checkpoints: List[WorkingMemoryCheckpoint] = []
    
    def start_state(
        self,
        state_name: str,
        content: Dict[str, Any],
        episode_id: str = "default",
        priority: int = 5,
        timestamp: Optional[str] = None
    ) -> str:
  
        node_id = self.trajectory.add_state(
            state_name=state_name,
            timestamp=timestamp or datetime.now().isoformat(),
            source_episode_id=episode_id,
            content=content,
            priority=priority
        )
        
        return node_id
    
    def transition_to(
        self,
        next_state_name: str,
        content: Dict[str, Any],
        reason: str = "",
        priority: int = 5,
        duration_seconds: Optional[int] = None,
        episode_id: str = "default",
        timestamp: Optional[str] = None
    ) -> str:
 
        current = self.trajectory.get_current_state()
        if current and duration_seconds:
            current.duration_seconds = duration_seconds

        new_node_id = self.trajectory.add_state(
            state_name=next_state_name,
            timestamp=timestamp or datetime.now().isoformat(),
            source_episode_id=episode_id,
            content=content,
            priority=priority,
            duration_seconds=None 
        )

        if self.trajectory.transitions:
            self.trajectory.transitions[-1].reason = reason
        
        return new_node_id
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
  
        node = self.trajectory.get_current_state()
        if not node:
            return None
        return {
            "state_id": node.node_id,
            "state_name": node.state_name,
            "progress": "In progress",
            "content": node.content,
            "priority": node.priority,
            "is_active": node.is_active
        }
    
    def get_execution_path(self) -> List[Dict[str, Any]]:

        result = []
        for ts, node_id in self.trajectory.timeline:
            node = self.trajectory.nodes.get(node_id)
            if node:
                result.append({
                    "state_id": node.node_id,
                    "state_name": node.state_name,
                    "timestamp": node.timestamp,
                    "duration_seconds": node.duration_seconds,
                    "priority": node.priority,
                    "content": node.content
                })
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        active_summary = self.trajectory.get_active_states_summary()
        stats = self.trajectory.get_statistics()
        total_states = stats["total_states"]
        progress = min(total_states / 10, 1.0)  
        
        capacity_usage = stats["capacity_usage"]
        if capacity_usage > 0.9:
            capacity_status = "Almost full"
        elif capacity_usage > 0.75:
            capacity_status = "Nearly full"
        else:
            capacity_status = "Normal"
        
        return {
            "task_name": self.task_name,
            "current_state": active_summary["current_state"],
            "total_states": total_states,
            "active_states": stats["active_states"],
            "archived_states": stats["archived_states"],
            "progress": round(progress, 2),
            "capacity_status": capacity_status,
            "capacity_usage": round(stats["capacity_usage"], 2),
            "state_timeline": self.get_execution_path(),
            "first_time": stats["first_time"],
            "last_time": stats["last_time"]
        }
    
    def get_state_transitions(self) -> List[Dict[str, Any]]:
        return self.trajectory.get_state_transitions()
    
    def get_incoming_transitions(self, node_id: str = None) -> List[Dict[str, Any]]:
        if node_id is None:
            current = self.trajectory.get_current_state()
            if current:
                node_id = current.node_id
            else:
                return []
        
        return self.trajectory.get_incoming_transitions(node_id)
    
    def get_outgoing_transitions(self, node_id: str = None) -> List[Dict[str, Any]]:
        if node_id is None:
            current = self.trajectory.get_current_state()
            if current:
                node_id = current.node_id
            else:
                return []
        
        return self.trajectory.get_outgoing_transitions(node_id)
    
    def save_checkpoint(self, checkpoint_name: str = "") -> str:
        current = self.trajectory.get_current_state()
        checkpoint_id = f"checkpoint_{len(self.checkpoints):04d}"
        
        checkpoint = WorkingMemoryCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            current_state_id=current.node_id if current else "",
            task_name=self.task_name,
            total_states=len(self.trajectory.nodes),
            states_snapshot={
                node_id: node for node_id, node in self.trajectory.nodes.items()
                if node.is_active
            },
            context=self.context.copy()
        )
        
        self.checkpoints.append(checkpoint)
        checkpoint_dir = self.storage_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{self.task_name}_checkpoint_{checkpoint_id}.json"

        data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp,
                "current_state_id": checkpoint.current_state_id,
                "task_name": checkpoint.task_name,
                "total_states": checkpoint.total_states,
                "context": checkpoint.context
            }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return checkpoint_id
    
    def get_status(self) -> str:
        summary = self.get_summary()
        current = summary["current_state"]
        
        if not current:
            return f"Task '{self.task_name}' has not started"
        
        return (
            f"Task: {self.task_name}\n"
            f"Current Stage: {current['state_name']}\n"
            f"Progress: {summary['progress']*100:.0f}%\n"
            f"States Completed: {summary['total_states']}\n"
            f"Active States: {summary['active_states']}/{summary['capacity_usage']*10:.0f}\n"
            f"Capacity Status: {summary['capacity_status']}"
        )
    
    def check_capacity(self) -> Dict[str, Any]:
        active_summary = self.trajectory.get_active_states_summary()
        
        return {
            "active_count": active_summary["active_count"],
            "capacity_limit": active_summary["capacity_limit"],
            "usage_percent": round(active_summary["capacity_usage"] * 100, 1),
            "needs_eviction": active_summary["capacity_usage"] > 0.8,
            "critical_states": active_summary["critical_states"]
        }


def create_working_memory(task_name: str, storage_dir: Optional[Path] = None) -> WorkingMemory:
    return WorkingMemory(task_name, storage_dir)
