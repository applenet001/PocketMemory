from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import json

from dotenv import load_dotenv
from memory.episodic_memory.episode_store import Episode

BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_LLM_MODEL = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")


def _call_llm(prompt: str, system_prompt: str = "") -> str:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=DEEPSEEK_LLM_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return ""


@dataclass
class ExtractedStateFromEpisodic:

    state_name: str                    
    state_category: str                 
    confidence: float                  

    progress_step: int = 0             
    total_steps: int = 0               
    progress_percentage: float = 0.0   
    

    recent_episodes: List[Episode] = None
    episode_chain_summary: str = ""

    current_focus: str = ""
    next_action_hint: str = ""

    priority: int = 5
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recent_episodes is None:
            self.recent_episodes = []
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_name": self.state_name,
            "state_category": self.state_category,
            "confidence": self.confidence,
            "progress_step": self.progress_step,
            "total_steps": self.total_steps,
            "progress_percentage": self.progress_percentage,
            "episode_chain_summary": self.episode_chain_summary,
            "current_focus": self.current_focus,
            "next_action_hint": self.next_action_hint,
            "priority": self.priority,
            "context": self.context,
        }


class StateExtractor:

    _state_cache: Dict[str, Dict[str, Any]] = {}
    _cache_max_size: int = 100 
    
    @staticmethod
    def _get_cache_key(recent_episodes: List[Episode]) -> str:
        if not recent_episodes:
            return "empty"
        return recent_episodes[-1].id
    
    @staticmethod
    def extract_state(
        recent_episodes: List[Episode],
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ExtractedStateFromEpisodic:
        if not recent_episodes:
            return StateExtractor._create_idle_state()

        cache_key = StateExtractor._get_cache_key(recent_episodes)
        if use_cache and cache_key in StateExtractor._state_cache:
            cached_analysis = StateExtractor._state_cache[cache_key]

            state = ExtractedStateFromEpisodic(
                state_name=cached_analysis.get("state_name", "Unknown status"),
                state_category=cached_analysis.get("state_category", "conversation"),
                confidence=cached_analysis.get("confidence", 0.7),
                progress_step=cached_analysis.get("current_step", 1),
                total_steps=cached_analysis.get("total_steps", 5),
                progress_percentage=cached_analysis.get("percentage", 20.0),
                recent_episodes=recent_episodes,
                episode_chain_summary=cached_analysis.get("summary", ""),
                current_focus=cached_analysis.get("focus", ""),
                next_action_hint=cached_analysis.get("next_action", ""),
                priority=cached_analysis.get("priority", 5),
                context=context or {},
            )
            return state
        
        analysis = StateExtractor._analyze_with_llm(recent_episodes)

        if use_cache:
            StateExtractor._state_cache[cache_key] = analysis

            if len(StateExtractor._state_cache) > StateExtractor._cache_max_size:
                oldest_key = next(iter(StateExtractor._state_cache))
                del StateExtractor._state_cache[oldest_key]

        state = ExtractedStateFromEpisodic(
            state_name=analysis.get("state_name", "Unknown status"),
            state_category=analysis.get("state_category", "conversation"),
            confidence=analysis.get("confidence", 0.7),
            progress_step=analysis.get("current_step", 1),
            total_steps=analysis.get("total_steps", 5),
            progress_percentage=analysis.get("percentage", 20.0),
            recent_episodes=recent_episodes,
            episode_chain_summary=analysis.get("summary", ""),
            current_focus=analysis.get("focus", ""),
            next_action_hint=analysis.get("next_action", ""),
            priority=analysis.get("priority", 5),
            context=context or {},
        )
        
        return state
    
    @staticmethod
    def clear_cache():
        StateExtractor._state_cache.clear()
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        return {
            "cached_states": len(StateExtractor._state_cache),
            "max_size": StateExtractor._cache_max_size,
            "usage_percentage": (len(StateExtractor._state_cache) / StateExtractor._cache_max_size) * 100,
        }
    
    @staticmethod
    def _analyze_with_llm(recent_episodes: List[Episode]) -> Dict[str, Any]:

        episode_texts = []
        for i, ep in enumerate(recent_episodes[-5:], 1):
            episode_texts.append(f"{i}. {ep.content}")
        
        episodes_str = "\n".join(episode_texts)
        
        # LLM Prompt
        system_prompt = "You are an AI assistant that specializes in analyzing users' behavior patterns to infer their current state and intentions."
        
        user_prompt = f"""Please analyze the following user behavior episode chain and infer the user's current state.
        Inference rules: When a user is in a particular event scenario, deduce their current specific state, e.g.:
        Please help me look up flights to Paris--->State: Looking up flights
        If the state cannot be determined, do not output anything.

Event Chain (in chronological order):
{episodes_str}

Please infer the following information based on the event chain and output strictly in JSON format:

{{
  "state_name": "State Name (2-6 characters, concise and specific)",
  "state_category": "State Category (one of task/query/conversation)",
  "current_step": Current Step Number (integer),
  "total_steps": 总步骤数（整数），
  "percentage": "Completion Percentage (float)",
  "focus": "Current Focus (short description of what the user is currently focused on)",
  "next_action": "Next Action Hint (short description of what the user might do next)",
  "priority": "Priority (integer from 1-10, higher means more urgent)",
  "summary": "Episode Chain Summary (one sentence summarizing the entire process)",
  "confidence": "Confidence (float from 0.0-1.0)"
}}

Requirements:
1. The `state_name` must be concise, intuitive, and easy to understand.
2. Output only JSON; do not include any other text.
3. All fields must be filled in."""

        try:
            response = _call_llm(user_prompt, system_prompt)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            analysis = json.loads(response)
            
            return analysis
            
        except Exception:
            return StateExtractor._fallback_analysis(recent_episodes)
    
    @staticmethod
    def _fallback_analysis(recent_episodes: List[Episode]) -> Dict[str, Any]:
        latest = recent_episodes[-1]
        
        return {
            "state_name": latest.content[:6] if len(latest.content) > 6 else latest.content,
            "state_category": "conversation",
            "current_step": len(recent_episodes),
            "total_steps": 10,
            "percentage": (len(recent_episodes) / 10) * 100,
            "focus": "User Input",
            "next_action": "Waiting for User Feedback",
            "priority": 5,
            "summary": "User is engaged in conversation",
            "confidence": 0.5,
        }
    
    @staticmethod
    def _create_idle_state() -> ExtractedStateFromEpisodic:
        return ExtractedStateFromEpisodic(
            state_name="Idle",
            state_category="conversation",
            confidence=1.0,
            progress_step=0,
            total_steps=0,
            progress_percentage=0.0,
            episode_chain_summary="No active events",
            current_focus="Waiting for input",
            next_action_hint="Waiting for user instruction",
            priority=1,
        )
