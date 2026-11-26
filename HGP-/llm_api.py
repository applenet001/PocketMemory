import os
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


class LLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        request_timeout: Optional[float] = None,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.request_timeout = request_timeout if request_timeout is not None else float(os.environ.get("LLM_REQUEST_TIMEOUT", "45"))

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call_llm(
        self,
        prompt: str,
        temperature: float = 0.2,
        system_prompt: str = "You are a helpful assistant.",
        request_timeout: Optional[float] = None,
    ) -> str:
        
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                timeout=request_timeout if request_timeout is not None else self.request_timeout,
            )
        return response.choices[0].message.content.strip()


    def call_llm_with_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        system_prompt: str = "You are a helpful assistant.",
        request_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:

        response_text = self.call_llm(prompt, temperature, system_prompt, request_timeout=request_timeout)
        if response_text.strip().startswith("```"):
            lines = response_text.strip().split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        import re
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)

        
        return json.loads(response_text)


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client