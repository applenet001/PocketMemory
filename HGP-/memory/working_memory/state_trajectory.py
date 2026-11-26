import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import heapq
import math


WORKING_MEMORY_CAPACITY = 7
WORKING_MEMORY_CAPACITY_MAX = 9


@dataclass
class StateNode:
    node_id: str                       
    state_name: str                   
    timestamp: str                     
    duration_seconds: Optional[int]     
    source_episode_id: str              
    event_id: str                     
    content: Dict[str, Any]             
    priority: int = 5                  
    access_count: int = 0              
    last_access: str = field(default_factory=lambda: datetime.now().isoformat())  
    is_active: bool = True              
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict) -> "StateNode":
        return StateNode(**data)
    
    def compute_importance_score(self) -> float:

        try:
            priority_score = min(self.priority / 10.0, 1.0)

            last_access_time = datetime.fromisoformat(self.last_access)
            seconds_passed = (datetime.now() - last_access_time).total_seconds()
            recency_score = math.exp(-seconds_passed / 3600)

            frequency_score = min(self.access_count / 10.0, 1.0)

            importance = (
                priority_score * 0.5 + 
                recency_score * 0.3 + 
                frequency_score * 0.2
            )
            
            return importance
        except Exception:
            return 0.5
    
    def update_access(self):
        self.access_count += 1
        self.last_access = datetime.now().isoformat()


@dataclass
class StateTransition:
    from_node_id: str
    to_node_id: str
    timestamp: str                    
    reason: str                      
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class StateTrajectoryGraph:
    def __init__(self, storage_path: Optional[Path] = None, event_id: str = "default", capacity: int = WORKING_MEMORY_CAPACITY):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "graph" / "working_memory" / "state_trajectory.json"
            storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.storage_path = Path(storage_path)
        self.event_id = event_id

        self.nodes: Dict[str, StateNode] = {}          
        self.transitions: List[StateTransition] = []   
        self.timeline: List[Tuple[str, str]] = []      

        self.incoming_transitions_map: Dict[str, List[StateTransition]] = {}  # to_node_id -> [transitions]
        self.outgoing_transitions_map: Dict[str, List[StateTransition]] = {}  # from_node_id -> [transitions]

        self.archived_nodes: Dict[str, StateNode] = {}  

        self._load_from_file()
    
    def _check_capacity(self) -> bool:
        active_count = sum(1 for node in self.nodes.values() if node.is_active)
        return active_count >= WORKING_MEMORY_CAPACITY_MAX
    
    def _evict_lowest_priority_state(self):
        if not self.nodes or not self.timeline:
            return

        current_node_id = self.timeline[-1][1] if self.timeline else None
        candidates = [
            (node_id, node) 
            for node_id, node in self.nodes.items() 
            if node.is_active and node_id != current_node_id
        ]
        
        if not candidates:
            return

        completed = [
            (nid, node) for nid, node in candidates 
            if node.content.get("status") == "completed"
        ]
        if completed:
            node_id, node = completed[0]
        else:
            paused = [
                (nid, node) for nid, node in candidates 
                if node.content.get("status") == "paused"
            ]
            if paused:
                node_id, node = paused[0]
            else:
                candidates_by_time = sorted(candidates, key=lambda x: x[1].timestamp)
                node_id, node = candidates_by_time[0]
        
        node.is_active = False
        self.archived_nodes[node_id] = node
    
    def add_state(
        self,
        state_name: str,
        timestamp: str,
        source_episode_id: str,
        content: Dict[str, Any],
        priority: int = 5,
        duration_seconds: Optional[int] = None
    ) -> str:
        if self._check_capacity():
            self._evict_lowest_priority_state()

        node_id = f"state_{len(self.nodes):04d}"

        node = StateNode(
            node_id=node_id,
            state_name=state_name,
            timestamp=timestamp,
            duration_seconds=duration_seconds,
            source_episode_id=source_episode_id,
            event_id=self.event_id,  
            content=content,
            priority=priority,
            is_active=True
        )

        if self.timeline:
            prev_timestamp, prev_node_id = self.timeline[-1]
            transition = StateTransition(
                from_node_id=prev_node_id,
                to_node_id=node_id,
                timestamp=timestamp,
                reason="自动转移"
            )
            self.transitions.append(transition)

            if node_id not in self.incoming_transitions_map:
                self.incoming_transitions_map[node_id] = []
            self.incoming_transitions_map[node_id].append(transition)
            
            if prev_node_id not in self.outgoing_transitions_map:
                self.outgoing_transitions_map[prev_node_id] = []
            self.outgoing_transitions_map[prev_node_id].append(transition)

        self.nodes[node_id] = node
        self.timeline.append((timestamp, node_id))

        self._save_node_to_file(node)
        
        return node_id
    
    def query_at_time(self, timestamp: str) -> Optional[StateNode]:

        best_node = None
        best_time = None
        for ts, node_id in self.timeline:
            if ts <= timestamp:
                best_time = ts
                best_node = node_id
            else:
                break
        
        if best_node:
            node = self.nodes.get(best_node)
            if node:
                node.update_access()
                return node
        
        return None
    
    def query_time_range(
        self,
        start_time: str,
        end_time: str
    ) -> List[StateNode]:
        
        result = []
        for ts, node_id in self.timeline:
            if start_time <= ts <= end_time:
                node = self.nodes.get(node_id)
                if node:
                    result.append(node)
        
        return result
    
    def get_path(
        self,
        start_node_id: str,
        end_node_id: str
    ) -> List[StateNode]:
        
        path = []
        current_id = start_node_id
        visited = set()

        max_iterations = len(self.nodes) + 1
        iterations = 0
        
        while current_id and iterations < max_iterations:
            iterations += 1

            if current_id in visited:
                break
            
            visited.add(current_id)
            node = self.nodes.get(current_id)
            
            if node:
                path.append(node)
            
            if current_id == end_node_id:
                break
            next_id = None
            for transition in self.transitions:
                if transition.from_node_id == current_id:
                    next_id = transition.to_node_id
                    break
            
            current_id = next_id
        
        return path
    
    def get_duration_by_state(self, state_name: str) -> int:
        total_duration = 0
        for node in self.nodes.values():
            if node.state_name == state_name and node.duration_seconds:
                total_duration += node.duration_seconds
        
        return total_duration
    
    def get_statistics(self) -> Dict[str, Any]:
        state_distribution = {}
        state_durations = {}
        state_importance_scores = {}
        
        active_count = 0
        archived_count = 0
        
        for node in self.nodes.values():
            state_distribution[node.state_name] = state_distribution.get(node.state_name, 0) + 1
            if node.duration_seconds:
                state_durations[node.state_name] = (
                    state_durations.get(node.state_name, 0) + node.duration_seconds
                )
            if node.is_active:
                active_count += 1
            else:
                archived_count += 1

            state_importance_scores[node.node_id] = {
                "state_name": node.state_name,
                "importance_score": node.compute_importance_score(),
                "priority": node.priority,
                "access_count": node.access_count
            }
        
        capacity_usage = active_count / self.capacity if self.capacity > 0 else 0
        
        return {
            "total_states": len(self.nodes),
            "active_states": active_count,
            "archived_states": archived_count,
            "total_transitions": len(self.transitions),
            "state_distribution": state_distribution,
            "first_time": self.timeline[0][0] if self.timeline else None,
            "last_time": self.timeline[-1][0] if self.timeline else None,
            "state_durations": state_durations,
            "capacity_usage": round(capacity_usage, 3),
            "capacity_limit": self.capacity,
            "state_importance_scores": state_importance_scores,
        }
    
    def get_state_transitions(self) -> List[Dict[str, Any]]:
        transitions_info = []
        for transition in self.transitions:
            from_node = self.nodes.get(transition.from_node_id)
            to_node = self.nodes.get(transition.to_node_id)
            
            if from_node and to_node:
                transitions_info.append({
                    "from_state": from_node.state_name,
                    "to_state": to_node.state_name,
                    "from_id": transition.from_node_id,
                    "to_id": transition.to_node_id,
                    "timestamp": transition.timestamp,
                    "reason": transition.reason,
                    "from_priority": from_node.priority,
                    "to_priority": to_node.priority
                })
        
        return transitions_info
    
    def get_incoming_transitions(self, node_id: str) -> List[Dict[str, Any]]:
        incoming = []

        transitions = self.incoming_transitions_map.get(node_id, [])
        
        for transition in transitions:
            from_node = self.nodes.get(transition.from_node_id)
            to_node = self.nodes.get(transition.to_node_id)
            if from_node and to_node:
                incoming.append({
                    "from_state": from_node.state_name,
                    "from_id": transition.from_node_id,
                    "timestamp": transition.timestamp,
                    "reason": transition.reason
                })
        return incoming
    
    def get_outgoing_transitions(self, node_id: str) -> List[Dict[str, Any]]:
        outgoing = []

        transitions = self.outgoing_transitions_map.get(node_id, [])
        
        for transition in transitions:
            from_node = self.nodes.get(transition.from_node_id)
            to_node = self.nodes.get(transition.to_node_id)
            if from_node and to_node:
                outgoing.append({
                    "to_state": to_node.state_name,
                    "to_id": transition.to_node_id,
                    "timestamp": transition.timestamp,
                    "reason": transition.reason
                })
        return outgoing
    
    def _save_node_to_file(self, node: StateNode):
        all_events = {}
        if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    all_events = json.load(f)

        if self.event_id not in all_events:
                all_events[self.event_id] = {
                    "nodes": [],
                    "transitions": []
                }

        if isinstance(all_events[self.event_id], list):
                all_events[self.event_id] = {
                    "nodes": all_events[self.event_id],
                    "transitions": []
                }

        all_events[self.event_id]["nodes"] = [
                n for n in all_events[self.event_id]["nodes"] 
                if n.get("node_id") != node.node_id
            ]
        all_events[self.event_id]["nodes"].append(node.to_dict())

        all_events[self.event_id]["transitions"] = [
                t.to_dict() for t in self.transitions
            ]

        with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(all_events, f, ensure_ascii=False, indent=2)
    
    def _load_from_file(self):
        if not self.storage_path.exists():
            return
        
        
        with open(self.storage_path, "r", encoding="utf-8") as f:
                all_events = json.load(f)

        if self.event_id in all_events:
                event_data = all_events[self.event_id]
                
                if isinstance(event_data, list):
                    nodes_data = event_data
                    transitions_data = []
                else:
                    nodes_data = event_data.get("nodes", [])
                    transitions_data = event_data.get("transitions", [])

                for data in nodes_data:
                    
                    node = StateNode.from_dict(data)
                    self.nodes[node.node_id] = node
                    self.timeline.append((node.timestamp, node.node_id))

                for t_data in transitions_data:
                    
                    transition = StateTransition(**t_data)
                    self.transitions.append(transition)

                    if transition.to_node_id not in self.incoming_transitions_map:
                            self.incoming_transitions_map[transition.to_node_id] = []
                    self.incoming_transitions_map[transition.to_node_id].append(transition)
                        
                    if transition.from_node_id not in self.outgoing_transitions_map:
                            self.outgoing_transitions_map[transition.from_node_id] = []
                    self.outgoing_transitions_map[transition.from_node_id].append(transition)

        self.timeline.sort(key=lambda x: x[0])
    
    def export_visualization(self) -> Dict[str, Any]:
        
        nodes = [
            {
                "id": node.node_id,
                "label": node.state_name,
                "timestamp": node.timestamp,
                "priority": node.priority,
                "is_active": node.is_active,
                "content": node.content
            }
            for node in self.nodes.values()
        ]
        
        links = [
            {
                "source": t.from_node_id,
                "target": t.to_node_id,
                "reason": t.reason,
                "timestamp": t.timestamp
            }
            for t in self.transitions
        ]
        
        return {
            "nodes": nodes,
            "links": links,
            "timeline": list(self.timeline),
            "statistics": self.get_statistics()
        }
    
    def get_current_state(self) -> Optional[StateNode]:
        if self.timeline:
            node_id = self.timeline[-1][1]
            node = self.nodes.get(node_id)
            if node and node.is_active:
                return node
        return None
    
    def get_active_states_summary(self) -> Dict[str, Any]:

        active_states = [node for node in self.nodes.values() if node.is_active]
        active_states.sort(key=lambda x: x.compute_importance_score(), reverse=True)
        
        return {
            "current_state": self.get_current_state(),
            "active_count": len(active_states),
            "capacity_usage": len(active_states) / self.capacity if self.capacity > 0 else 0,
            "capacity_limit": self.capacity,
            "critical_states": [
                {
                    "node_id": node.node_id,
                    "state_name": node.state_name,
                    "priority": node.priority,
                    "importance_score": node.compute_importance_score()
                }
                for node in active_states[:3] 
            ]
        }


_default_trajectory = StateTrajectoryGraph()

