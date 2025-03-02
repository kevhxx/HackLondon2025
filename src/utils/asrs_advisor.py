import json
from typing import List

from openai import BaseModel
from src.client import client


class Suggestion(BaseModel):
    title: str
    content: str

    class Config:
        extra = 'forbid'  # Disallow additional properties


class StudyPlanSuggestion(BaseModel):
    study_session: float  # 学习时长，单位可以是分钟或小时
    break_interval: float  # 休息间隔，单位可以是分钟或小时
    sessions_per_day: int  # 每天的学习次数

    class Config:
        extra = 'forbid'  # 不允许额外的属性


class AdviceResponse(BaseModel):
    recommendations: List[Suggestion]
    study_plan: StudyPlanSuggestion

    class Config:
        extra = 'forbid'  # 不允许额外的属性


async def generate_advice(asrs_results: str):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                **Prompt:**
                As an ADHD expert, please provide personalized recommendations based on the given ASRS Test Results.
                
                The expected output is a JSON object matching the following structure:
                ```python
                class Suggestion(BaseModel):
                    title: str
                    content: str
                
                class StudyPlanSuggestion(BaseModel):
                    study_session: float  # 学习时长，单位是分钟
                    break_interval: float  # 休息间隔，是分钟
                    sessions_per_day: int  # 每天的学习次数
                
                
                class AdviceResponse(BaseModel):
                    recommendations: List[Suggestion]
                    study_plan: StudyPlanSuggestion
=
                Please limit the overall length of the content to 500 words. Ensure that the advice is clear, actionable, and tailored to the individual's responses.
                
                reply lang: en-us
                """
            },
            {"role": "user", "content": f"{asrs_results}"}
        ],
        response_format=AdviceResponse,
    )
    try:
        advice = completion.choices[0].message.parsed
    except Exception as e:
        print(f"Failed to generate advice: {str(e)}")
        # Handle exceptions or retry logic as needed
        await generate_advice(asrs_results)
    return advice


if __name__ == "__main__":
    import asyncio

    content = """
        Part A Score
        6/6
        Core ADHD symptoms
        
        Part B Score
        8/12
        Additional symptoms
        
        Total Score
        14/18
        Overall assessment
    """
    advice = asyncio.run(generate_advice(content))
    print(json.dumps(advice.dict(), indent=2))
