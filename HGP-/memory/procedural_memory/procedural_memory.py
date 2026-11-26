from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from llm_api import get_llm_client
from .store import ProcedureRule, ProceduralMemoryStore

llm_client = get_llm_client()
_default_store = ProceduralMemoryStore()


def extract_procedure_rules(text: str, meta: Optional[Dict[str, Any]] = None) -> List[ProcedureRule]:

    prompt = f"""
You are a “procedural memory” extraction assistant. You are now given a piece of text that may describe a workflow,
an “if...then...” rule, or a multi-step procedure. Extract (0 to N) reusable rules from the text that have received positive feedback from users. Each rule is represented as follows:
1. “title”: A short name for the process/rule
2. “if”: A list of trigger conditions (an array of strings; if not specified in the text, an empty array is acceptable)
3. “then”: A chronological list of “main steps” (each step described in a single short verb sentence)
4. “tags”: A set of short tags (e.g., [“programming”, “environment setup”], [‘lifestyle’, “cooking”])
5. “tree”: A tree-like structure capable of representing complex workflows, supporting both sequential and branching paths, formatted as follows:
- Sequential execution (sequence):
     {{
       “type”: “sequence”,
       “steps”: [ Node, Node, ... ]
     }}

   - Regular step (action):
     {{
       “type”: “action”,
       “text”: “Specific action to be performed”
     }}

   - Conditional branch (branch):
     {{
       “type”: “branch”,
       “condition”: “Under what conditions this branch is considered (e.g., installation failure)”,
       “cases”: [
         {{
           “when”: “Specific scenario A (e.g., network error)”,
           “steps”: [ Node, Node, ... ]
         }},
         {{
           “when”: “Specific scenario B (e.g., insufficient permissions)”,
           “steps”: [ Node, Node, ... ]
         }}
       ]
     }}

   Here, Node can be “action”, ‘branch’, or a nested “sequence”.
   If the workflow is simple and has no branches, you can also make the tree a simple sequence corresponding to each step of the `then` block:

   {{
     “type”: “sequence”,
     “steps”: [
       {{“type”: “action”, ‘text’: “Step 1”}},
       {{“type”: “action”, ‘text’: “Step 2”}}
     ]
   }}

Please output strictly according to the JSON structure below (do not include any additional explanations):

{{
  “rules”: [
    {{
      “title”: “Rule Name”,
      “if”: [“Condition 1”, “Condition 2”],
      “then”: [“Step 1”, “Step 2”, “Step 3”],
      “tags”: [“Tag 1”, “Tag 2”],
      “tree”: {{
        “type”: “...”,
        “steps”: [...]
      }}
    }}
  ]
}}

If the text does not contain any obvious workflow or rules, you may return an empty list: {{“rules”: []}}.

Here is the text:
{text}
    """.strip()

    
    data = llm_client.call_llm_with_json(
            prompt=prompt,
            temperature=0.2,
            system_prompt="You are a tool that helps extract reusable procedural rules (if-then statements) and flowcharts from text."
        )

    rules: List[ProcedureRule] = []
    raw_rules = data.get("rules", [])

    if not isinstance(raw_rules, list):
        return rules

    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title", "")).strip()
        if_list = [str(x).strip() for x in r.get("if", []) if str(x).strip()]
        then_list = [str(x).strip() for x in r.get("then", []) if str(x).strip()]
        tags = [str(x).strip() for x in r.get("tags", []) if str(x).strip()]

        if not then_list:
            continue
        
        raw_tree=r.get("tree")
        if isinstance(raw_tree,dict) and isinstance(raw_tree.get("type"),str):
            tree=raw_tree
        else:
            tree=ProcedureRule.build_tree(if_list,then_list)    
        rule = ProcedureRule(
            id="", 
            title=title or (then_list[0] if then_list else "Untitled Procedure"),
            if_conditions=if_list,
            then_steps=then_list,
            tree=tree,
            tags=tags,
            source_text=text,
            meta=meta or {},
        )
        rules.append(rule)

    return rules


def store_procedural_memory(text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rules = extract_procedure_rules(text, meta=meta)
    if not rules:
        return {
            "text": text,
            "rules": [],
            "stored_rule_ids": [],
        }

    store = _default_store
    ids = store.add_rules(rules)

    return {
        "text": text,
        "rules": [r.to_dict() for r in rules],
        "stored_rule_ids": ids,
    }

