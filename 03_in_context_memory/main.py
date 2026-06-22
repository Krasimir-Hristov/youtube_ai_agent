import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# OpenRouter Client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)
