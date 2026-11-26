from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json



class UserProfileCategory(str, Enum):
    FAMILY = "family"           
    HEALTH = "health"          
    WORK = "work"             
    FOOD = "food"             
    HOBBY = "hobby"             
    PERSONALITY = "personality" 
    SOCIAL = "social"          
    PREFERENCE = "preference"  
    BACKGROUND = "background"   


class SemanticNodeType(str, Enum):
    USER = "User"               
    FEATURE = "Feature"         
    ENTITY = "Entity"           
    CONCEPT = "Concept"         


class SemanticRelationType(str, Enum):
    HAS_FEATURE = "HAS_FEATURE"     
    LIKES = "LIKES"                
    DISLIKES = "DISLIKES"           
    KNOWS = "KNOWS"                 
    RELATED_TO = "RELATED_TO"      

class EpisodicNodeType(str, Enum):
    EVENT = "Event"            
    PERSON = "Person"          
    TIME = "Time"               
    LOCATION = "Location"      
    TOPIC = "Topic"          

class EpisodicRelationType(str, Enum):
    INVOLVES = "INVOLVES"       
    HAPPENS_AT = "HAPPENS_AT"   
    OCCURS_AT = "OCCURS_AT"     
    NEXT = "NEXT"               
    ABOUT = "ABOUT"          

@dataclass
class NodeSchema:
    type: str                      
    required_props: List[str]     
    optional_props: List[str]     
    description: str             
    

@dataclass
class RelationSchema:
    type: str                      
    source: str                    
    target: str                  
    description: str           

SEMANTIC_SCHEMAS = {
    "nodes": {
        SemanticNodeType.USER: NodeSchema(
            type=SemanticNodeType.USER,
            required_props=["user_id", "namespace"],
            optional_props=["name", "created_at", "updated_at"],
            description="User nodes, as the hub of user profiling"
        ),
        SemanticNodeType.FEATURE: NodeSchema(
            type=SemanticNodeType.FEATURE,
            required_props=["category", "content", "namespace"],
            optional_props=["confidence", "source", "evidence_episode_ids", "created_at"],
            description="Feature nodes, describing user attributes. evidence_episode_ids records source episode IDs for traceability"
        ),
        SemanticNodeType.ENTITY: NodeSchema(
            type=SemanticNodeType.ENTITY,
            required_props=["name", "type", "namespace"],
            optional_props=["description", "aliases"],
            description="Entity nodes, representing specific things "
        ),
        SemanticNodeType.CONCEPT: NodeSchema(
            type=SemanticNodeType.CONCEPT,
            required_props=["name", "namespace"],
            optional_props=["description", "category"],
            description="Concept node, representing abstract knowledge"
        ),
    },
    "relations": {
        SemanticRelationType.HAS_FEATURE: RelationSchema(
            type=SemanticRelationType.HAS_FEATURE,
            source=SemanticNodeType.USER,
            target=SemanticNodeType.FEATURE,
            description="Users who have a certain characteristic"
        ),
        SemanticRelationType.LIKES: RelationSchema(
            type=SemanticRelationType.LIKES,
            source=SemanticNodeType.USER,
            target=SemanticNodeType.ENTITY,
            description="Users who like a certain entity"
        ),
        SemanticRelationType.DISLIKES: RelationSchema(
            type=SemanticRelationType.DISLIKES,
            source=SemanticNodeType.USER,
            target=SemanticNodeType.ENTITY,
            description="Users who dislike a certain entity"
        ),
    }
}


EPISODIC_EXTRACTION_SCHEMA = {
    "type": "object",
    "description": "Extracting structured information from event descriptions",
    "properties": {
        "participants": {
            "type": "array",
            "description": "Event participants (people)",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Participant name"},
                    "role": {"type": "string", "description": "Role in the event (optional)"}
                },
                "required": ["name"]
            }
        },
        "persons": {
            "type": "array",
            "description": "All person names mentioned in the conversation (including participants and mentioned people)",
            "items": {"type": "string"}
        },
        "topics": {
            "type": "array",
            "description": "Topics, discussions, and key concepts involved in the conversation",
            "items": {"type": "string"}
        },
        "time": {
            "type": "object",
            "description": "Event occurrence time",
            "properties": {
                "value": {"type": "string", "description": "Time description (e.g., 'August 2023', 'yesterday', 'last summer')"},
                "year": {"type": "string", "description": "Year (e.g., '2023')"},
                "month": {"type": "string", "description": "Month (e.g., 'August')"},
                "day": {"type": "string", "description": "Date (e.g., '14')"}
            }
        },
        "location": {
            "type": "array",
            "description": "Event occurrence location",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Location name"},
                    "type": {"type": "string", "description": "Location type (e.g., 'park', 'museum', 'home')"}
                },
                "required": ["name"]
            }
        }
    },
    "required": ["participants"]
}


SEMANTIC_EXTRACTION_SCHEMA = {
    "type": "object",
    "description": "Extracting user profile information from text",
    "properties": {
        "features": {
            "type": "array",
            "description": "User features",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["family", "health", "work", "food", "hobby", "personality", "social", "preference", "background"],
                        "description": "Feature category"
                    },
                    "content": {"type": "string", "description": "Feature content description"}
                },
                "required": ["category", "content"]
            }
        },
        "likes": {
            "type": "array",
            "description": "User likes",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name"},
                    "type": {"type": "string", "description": "Entity type (e.g., 'food', 'activity', 'person')"}
                },
                "required": ["name"]
            }
        },
        "dislikes": {
            "type": "array",
            "description": "User dislikes",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name"},
                    "type": {"type": "string", "description": "Entity type (e.g., 'food', 'activity', 'person')"}
                },
                "required": ["name"]
            }
        }
    }
}


def get_episodic_extraction_prompt(content: str) -> str:
    return f"""You are an event information extraction assistant. Please extract structured event information from the following text.

**Extraction Guidelines**:

1. **participants**: People directly involved in or initiating the event
   - Includes speakers and action takers
   - 示例：讲故事的人、做决定的人、采取行动的人

2. **persons**: All people mentioned in the text
   - Includes participants and other mentioned individuals
   - Includes fictional characters, celebrities, group representatives, etc.
   - Example: All names mentioned directly

3. **topics (themes and activities)**: All themes, activities, and keywords related to the event
   - **Includes specific activities**: Such as going somewhere or doing something; these can be considered brief descriptive terms for the activity
   - **Includes concepts and topics**: Concepts and related topics involved in the event
   - **Includes objects and things**: such as books, dogs, cars, etc.
   - Avoid duplicating extracted topics; use as few topics as possible to cover the information in the event

4. **time (time information)**: The time the event occurred
   - Specific dates: Extract year, month, and day (e.g., 2023-08-15)
   - Relative time: Keep the original wording (e.g., “last Friday,” “August,” “yesterday,” “last week”)
   - Format: {{“value”: “time description”, “granularity”: ‘absolute’ or “relative”}}

5. **location (Location)**: The location where the event occurred
   - Includes specific locations: cities, buildings, parks, etc.
   - Includes conceptual locations: home, office, outdoors, etc.

6. **Extraction Principles**:
   - Extract only information **explicitly mentioned** in the text
   - When encountering uncertain names or activities, **lean toward inclusion rather than exclusion**
   - If a piece of information is missing, return an empty array or empty object

**Example**:

Input：Lily: I went to the museum with my family last Sunday. We saw the dinosaur exhibit and discussed paleontology. The kids loved it.

Output：
{{
  "participants": ["Lily"],
  "persons": ["Lily", "family", "kids"],
  "topics": ["museum visit", "dinosaur", "paleontology", "family outing"],
  "time": {{
    "value": "last Sunday",
    "granularity": "relative"
  }},
  "location": [
    {{"name": "museum", "type": "museum"}}
  ]
}}

Input：Sarah mentioned she had a picnic at the beach yesterday with Tom. They brought sandwiches and played volleyball.

Output：
{{
  "participants": ["Sarah"],
  "persons": ["Sarah", "Tom"],
  "topics": ["picnic", "beach", "volleyball", "sandwiches", "outdoor activity"],
  "time": {{
    "value": "yesterday",
    "granularity": "relative"
  }},
  "location": [
    {{"name": "beach", "type": "beach"}}
  ]
}}

---

Please extract the following content:

**Text content**:
{content}

**Please output the extraction results in JSON format** (ensure that the topics include all specific activities, things, and concepts mentioned)："""


def get_semantic_extraction_prompt(content: str) -> str:

    return f"""You are a user profiling assistant. Please extract the user’s characteristics and preferences from the text below.

**Extraction Requirements**:
1. **features**: Extract characteristics that describe the user and categorize them by type
   - family: family members, marital status, children, etc.
   - health: physical condition, exercise habits, etc.
   - **Work**: Occupation, company, skills, etc.
   - **Food**: Favorite foods, eating habits, etc.
   - **Hobbies**: Interests, activities, etc.
   - **Personality**: Personality traits, values, etc.
   - **Social**: Friends, social circles, etc.
   - **Preferences**: General preferences
   - **Background**: Education, experience, etc.

2. **likes**: Things the user has explicitly indicated they like
3. **dislikes**: Things the user has explicitly indicated they dislike

**Example**:

Input：Caroline is a transgender woman who works as a counselor. She has been passionate about LGBTQ+ rights. She loves running and painting.

Output：
{{
  "features": [
    {{"category": "background", "content": "is a transgender woman"}},
    {{"category": "work", "content": "works as a counselor"}},
    {{"category": "personality", "content": "passionate about LGBTQ+ rights"}}
  ],
  "likes": [
    {{"name": "running", "type": "activity"}},
    {{"name": "painting", "type": "activity"}}
  ],
  "dislikes": []
}}

---

Please extract the following content:

**Text content**:
{content}

**Please output the extraction results in JSON format**："""

