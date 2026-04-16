import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI


load_dotenv()


def _clean(value: str | None) -> str:
    return (value or '').strip()


def _build_model():
    model_source = _clean(os.getenv('model_source', 'azure_openai')).lower()

    # Keep Azure explicit to avoid configurable-model edge cases introduced by
    # newer langchain versions when model/provider resolution is ambiguous.
    if model_source == 'azure_openai':
        deployment = _clean(os.getenv('CHAT_DEPLOYMENT_NAME'))
        raw_model = _clean(os.getenv('CHAT_MODEL_NAME'))
        model_name = raw_model.split(':', 1)[-1] if ':' in raw_model else raw_model
        model_name = model_name or deployment or 'gpt-4.1'

        return AzureChatOpenAI(
            azure_deployment=deployment or model_name,
            azure_endpoint=_clean(os.getenv('AZURE_OPENAI_ENDPOINT')),
            api_key=_clean(os.getenv('AZURE_OPENAI_API_KEY')),
            api_version=_clean(os.getenv('OPENAI_API_VERSION')) or '2024-12-01-preview',
            model=model_name,
        )

    model_name = _clean(os.getenv('CHAT_MODEL_NAME')) or 'gpt-4.1'
    return init_chat_model(model=model_name)


llm = _build_model()
