import json
from openai import BaseModel
from src.client import client



class Blocks(BaseModel):
    title: str
    content: str
    start_time: str
    end_time: str

    class Config:
        extra = 'forbid'  # Disallow additional properties


class EventResponse(BaseModel):
    number: int
    start_time: str
    end_time: str
    overall_duration: str
    blocks: list[Blocks]

    class Config:
        extra = 'forbid'  # Disallow additional properties


async def generate_time_table(content: str):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
            **Prompt:**
                As an ADHD expert, please arrange the tasks for each study session based on the input provided for "Study Session Break Interval" and "Sessions per Day". Your task is to schedule the corresponding tasks for each session.
                The expected output is a JSON object matching the following structure:
                ```python
                class Blocks(BaseModel):
                    title: str
                    content: str
                    start_time: str
                    end_time: str
                
                class EventResponse(BaseModel):
                    number: int  # Total number of sessions
                    start_time: str  # Start time of the first session
                    end_time: str  # End time of the last session
                    overall_duration: str  # Total duration of all sessions
                    blocks: list[Blocks]  # List of all scheduled blocks
                ```
                Please ensure that each task is assigned appropriately, considering the break intervals and total sessions per day. Provide accurate start and end times for each block.
                
                reply lang: ENGLISH
                """},
            {"role": "user", "content": f"{content}"}
        ],
        response_format=EventResponse,
    )
    try:
        event = completion.choices[0].message.parsed
    except Exception as e:
        await generate_time_table(content)
    return event


if __name__ == '__main__':
    content = ("""
                Study Session
                25 min
                Break Interval
                8 min
                08-09: 自习
                10-12: 作业
                12-14 复习
                14-16: 复习
                """)
    import asyncio

    event = asyncio.run(generate_time_table(content))
    try:
        json.loads(event.model_dump_json())
        # print formatted json
        print(json.dumps(json.loads(event.model_dump_json()), indent=4))
    except Exception as e:
        print(e)
        print(event.model_dump_json())
