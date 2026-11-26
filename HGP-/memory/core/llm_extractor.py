#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI


def clean_json_response(text: str) -> str:
    if not text:
        return text

    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    text = text.strip()
    
    return text

from .unified_schema import (
    get_episodic_extraction_prompt,
    get_semantic_extraction_prompt,
    EPISODIC_EXTRACTION_SCHEMA,
    SEMANTIC_EXTRACTION_SCHEMA
)


class SchemaBasedExtractor:
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):

        if base_url is None:
            base_url = "https://api.openai.com/v1"
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
    
    def extract_from_episode(
        self,
        episode_content: str,
        namespace: str = "default"
    ) -> Dict[str, Any]:

        prompt = get_episodic_extraction_prompt(episode_content)
        
        try:
            request_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise knowledge graph extraction assistant. Output structured data strictly in accordance with the JSON Schema. Output only JSON; do not include any other content."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
            }

            if "deepseek" not in self.base_url.lower():
                request_params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**request_params)
            
            result_text = response.choices[0].message.content

            if not result_text or not result_text.strip():
                print(f"[SchemaExtractor] The LLM returns an empty response")
                return {
                    "participants": [],
                    "time": None,
                    "location": [],
                    "namespace": namespace
                }

            result_text = clean_json_response(result_text)

            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                return {
                    "participants": [],
                    "time": None,
                    "location": [],
                    "namespace": namespace
                }

            result["namespace"] = namespace
            
            return self._validate_episodic_result(result)
            
        except Exception:
            return {
                "participants": [],
                "time": None,
                "location": [],
                "namespace": namespace
            }
    
    def extract_from_semantic(
        self,
        content: str,
        namespace: str = "default"
    ) -> Dict[str, Any]:

        prompt = get_semantic_extraction_prompt(content)
        
        try:
            request_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise knowledge graph extraction assistant. Output structured data strictly in accordance with the JSON Schema. Output only JSON; do not include any other content."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
            }

            if "deepseek" not in self.base_url.lower():
                request_params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**request_params)
            
            result_text = response.choices[0].message.content

            if not result_text or not result_text.strip():
                return {
                    "features": [],
                    "likes": [],
                    "dislikes": [],
                    "namespace": namespace
                }
            
            result_text = clean_json_response(result_text)
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                return {
                    "features": [],
                    "likes": [],
                    "dislikes": [],
                    "namespace": namespace
                }
            
            result["namespace"] = namespace
            return self._validate_semantic_result(result)
        
        except Exception:
            return {
                "features": [],
                "likes": [],
                "dislikes": [],
                "namespace": namespace
            }
    
    def _validate_episodic_result(self, result: Dict) -> Dict:
        result.setdefault("participants", [])
        result.setdefault("time", None)
        result.setdefault("location", [])
        validated_participants = []
        for p in result.get("participants", []):
            if isinstance(p, dict) and "name" in p:
                validated_participants.append({
                    "name": p["name"],
                    "role": p.get("role", "participant")
                })
            elif isinstance(p, str):
                validated_participants.append({"name": p, "role": "participant"})
        result["participants"] = validated_participants

        if result.get("time") and isinstance(result["time"], dict):
            if "value" not in result["time"]:
                result["time"] = None

        validated_location = []
        for loc in result.get("location", []):
            if isinstance(loc, dict) and "name" in loc:
                validated_location.append({
                    "name": loc["name"],
                    "type": loc.get("type", "location")
                })
            elif isinstance(loc, str):
                validated_location.append({"name": loc, "type": "location"})
        result["location"] = validated_location
        
        return result
    
    def _validate_semantic_result(self, result: Dict) -> Dict:
 
        result.setdefault("features", [])
        result.setdefault("likes", [])
        result.setdefault("dislikes", [])
 
        validated_features = []
        for feature in result.get("features", []):
            if isinstance(feature, dict) and "category" in feature and "content" in feature:
                validated_features.append({
                    "category": feature["category"],
                    "content": feature["content"]
                })
        result["features"] = validated_features

        validated_likes = []
        for like in result.get("likes", []):
            if isinstance(like, dict) and "name" in like:
                validated_likes.append({
                    "name": like["name"],
                    "type": like.get("type", "entity")
                })
            elif isinstance(like, str):
                validated_likes.append({"name": like, "type": "entity"})
        result["likes"] = validated_likes

        validated_dislikes = []
        for dislike in result.get("dislikes", []):
            if isinstance(dislike, dict) and "name" in dislike:
                validated_dislikes.append({
                    "name": dislike["name"],
                    "type": dislike.get("type", "entity")
                })
            elif isinstance(dislike, str):
                validated_dislikes.append({"name": dislike, "type": "entity"})
        result["dislikes"] = validated_dislikes
        
        return result
    
    
    def extract_semantic_info(self, content: str, namespace: str = "default") -> Dict[str, Any]:
        return self.extract_from_semantic(content, namespace)
    
    def extract_episodic_info(self, content: str, namespace: str = "default") -> Dict[str, Any]:
        return self.extract_from_episode(content, namespace)
    
    def batch_extract_from_episodes(
        self,
        episodes: List[Dict[str, Any]],
        namespace: str = "default"
    ) -> List[Dict[str, Any]]:
        results = []
        for i, episode in enumerate(episodes):
            content = episode.get("content", "")
            if not content:
                continue
            
            extracted = self.extract_from_episode(content, namespace)
            extracted["episode_id"] = episode.get("id")
            results.append(extracted)
        
        return results
