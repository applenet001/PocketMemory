#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class AdaptiveThresholdManager:
    def __init__(
        self,
        tau_0: float = 0.9,     
        tau_min: float = 0.5,    
        beta: float = 0.05,      
        config_path: Optional[Path] = None
    ):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.beta = beta

        self.config_path = config_path or Path("train/threshold_config.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "iterations": 0,        
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "history": []            
        }
    
    def _save_state(self):
        self.state["last_updated"] = datetime.now().isoformat()
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
    
    def get_current_threshold(self) -> float:
        t = self.state["iterations"]
        tau_t = self.tau_0 * math.exp(-self.beta * t)
        tau_t = max(self.tau_min, tau_t)
        
        return round(tau_t, 4)
    
    def record_training_iteration(self, num_samples: Optional[int] = None):

        old_threshold = self.get_current_threshold()

        self.state["iterations"] += 1
        
        new_threshold = self.get_current_threshold()

        self.state["history"].append({
            "iteration": self.state["iterations"],
            "threshold": new_threshold,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_state()
    
    def reset(self):
        old_iterations = self.state["iterations"]
        old_threshold = self.get_current_threshold()

        self.state["iterations"] = 0
        self.state["history"].append({
            "event": "reset",
            "old_iterations": old_iterations,
            "old_threshold": old_threshold,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_state()
    
    def get_stats(self) -> Dict[str, Any]:
        current_threshold = self.get_current_threshold()

        threshold_reduction = (self.tau_0 - current_threshold) / self.tau_0 * 100
        if current_threshold > self.tau_min:
            t_max = -math.log(self.tau_min / self.tau_0) / self.beta
            remaining_iterations = int(t_max - self.state["iterations"])
        else:
            remaining_iterations = 0
        
        return {
            "current_iterations": self.state["iterations"],
            "current_threshold": current_threshold,
            "initial_threshold": self.tau_0,
            "min_threshold": self.tau_min,
            "threshold_reduction_percent": round(threshold_reduction, 2),
            "remaining_iterations_to_min": max(0, remaining_iterations),
            "decay_rate": self.beta,
            "total_history_records": len(self.state["history"])
        }
    
_default_manager = None


def get_threshold_manager() -> AdaptiveThresholdManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AdaptiveThresholdManager()
    return _default_manager


def get_current_classifier_threshold() -> float:
    manager = get_threshold_manager()
    return manager.get_current_threshold()


def record_training() -> float:
    manager = get_threshold_manager()
    manager.record_training_iteration()
    return manager.get_current_threshold()


def reset_classifier() -> float:
    manager = get_threshold_manager()
    manager.reset()
    return manager.get_current_threshold()
