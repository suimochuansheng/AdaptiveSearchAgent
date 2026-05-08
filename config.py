from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    tavily_api_key: str
    deepseek_api_key: str = ""
    OLLAMA_BASE_URL: str = ""
    OPENAI_BASE_URL: str = ""
    MODEL_NAME: str = ""
    OPENAI_API_KEY: str = ""
    # 其他配置...

    class Config:
        env_file = ".env"  # 自动加载项目根目录的 .env
        env_file_encoding = "utf-8"


settings = Settings()
