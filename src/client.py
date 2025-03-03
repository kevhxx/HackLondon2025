from logging import raiseExceptions

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
# client = OpenAI(
#     api_key="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")

# dont worry about the api key, it's a public key
# controlled by @binaryyuki's server and it's not a secret key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else print("OPENAI_API_KEY not found"),
    base_url=os.getenv("OPENAI_BASE_URL") if os.getenv("OPENAI_BASE_URL") else print("OPENAI_BASE_URL not found")
)


def get_client():
    """
    Get the OpenAI client
    :return:
    """
    return client
