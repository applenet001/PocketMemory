from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

from .state_trajectory import (
    StateTrajectoryGraph,
    StateNode,
    StateTransition,
    WORKING_MEMORY_CAPACITY,
    WORKING_MEMORY_CAPACITY_MAX
)
from .working_memory import (
    WorkingMemory,
    WorkingMemoryCheckpoint,
    create_working_memory
)


class StateType(Enum):
    """工作记忆状态类型"""
    PENDING = "pending"          
    EXECUTING = "executing"      
    WAITING = "waiting"          
    COMPLETED = "completed"      
    FAILED = "failed"           
    PAUSED = "paused"          

_working_memory_instances: Dict[str, WorkingMemory] = {}


def _get_working_memory(fsm_name: str = "main_task") -> WorkingMemory:
    if fsm_name not in _working_memory_instances:
        _working_memory_instances[fsm_name] = create_working_memory(fsm_name)
    return _working_memory_instances[fsm_name]

def store_working_state(
    state_name: str,
    state_type: StateType = StateType.EXECUTING,
    context: Optional[Dict[str, Any]] = None,
    fsm_name: str = "main_task",
    priority: int = 5,
) -> Dict[str, Any]:
    try:
        wm = _get_working_memory(fsm_name)

        content = {
            "state_type": state_type.value,
            **(context or {}),
        }

        node_id = wm.start_state(
            state_name=state_name,
            content=content,
            priority=priority,
        )
        
        return {
            "success": True,
            "node_id": node_id,
            "state_name": state_name,
            "state_type": state_type.value,
            "fsm_name": fsm_name,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "state_name": state_name,
        }


def query_working_state(
    fsm_name: str = "main_task",
) -> Optional[Dict[str, Any]]:
    try:
        wm = _get_working_memory(fsm_name)
        return wm.get_current_state()
    except Exception:
        return None


def get_working_memory_summary(
    fsm_name: str = "main_task",
) -> Dict[str, Any]:

    try:
        wm = _get_working_memory(fsm_name)
        return wm.get_summary()
    except Exception as e:
        return {"error": str(e)}


__all__ = [
    "StateTrajectoryGraph",
    "StateNode",
    "StateTransition",

    "WorkingMemory",
    "WorkingMemoryCheckpoint",
    "create_working_memory",

    "store_working_state",
    "query_working_state",
    "get_working_memory_summary",
    "StateType",
 
    "WORKING_MEMORY_CAPACITY",
    "WORKING_MEMORY_CAPACITY_MAX",
]
