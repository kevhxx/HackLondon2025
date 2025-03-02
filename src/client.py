from openai import OpenAI

# client = OpenAI(
#     api_key="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")

client = OpenAI(
    api_key="sk-U7WAlocyRe8nagQcqG1uXzRUKyU8PnUc4OHY6rGU8fENVR8F",
    base_url="https://ai.tzpro.xyz/v1"
)

def get_client():
    """
    Get the OpenAI client
    :return:
    """
    return client
