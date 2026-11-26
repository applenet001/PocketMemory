import os
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:

    OPENAI_BASE_URL = "https://api.openai.com/v1"
    OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder"]

    OTHER_BASE_URLS = {
        "ollama": "http://localhost:11434/v1",
        "local": "http://localhost:8000/v1",
    }
    
    @staticmethod
    def get_deepseek_config() -> Tuple[str, str, str]:
        api_key = os.getenv("DEEPSEEK_API_KEY")        
        base_url = os.getenv("DEEPSEEK_API_BASE", LLMConfig.DEEPSEEK_BASE_URL)
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        return api_key, base_url, model
    
    @staticmethod
    def get_openai_config() -> Tuple[str, str, str]:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", LLMConfig.OPENAI_BASE_URL)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        return api_key, base_url, model
    
    @staticmethod
    def get_llm_config(provider: str = "auto") -> Tuple[str, str, str]:
        if provider == "auto":
            if os.getenv("DEEPSEEK_API_KEY"):
                return LLMConfig.get_deepseek_config()
            elif os.getenv("OPENAI_API_KEY"):
                return LLMConfig.get_openai_config()
        elif provider == "deepseek":
            return LLMConfig.get_deepseek_config()
        
        elif provider == "openai":
            return LLMConfig.get_openai_config()
    @staticmethod
    def print_config_help():
        '''Print configuration help'''


def get_default_llm_config() -> Tuple[str, str, str]:
    return LLMConfig.get_llm_config("auto")


def get_deepseek() -> Tuple[str, str, str]:
    return LLMConfig.get_deepseek_config()


def get_openai() -> Tuple[str, str, str]:
    return LLMConfig.get_openai_config()


if __name__ == "__main__":
    LLMConfig.print_config_help()
    api_key, base_url, model = get_default_llm_config()

