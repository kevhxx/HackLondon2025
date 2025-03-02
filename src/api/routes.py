import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from fastapi.requests import Request
from src.models.Diagnose import InputData
from src.models.study_plan import StudyPlanRequest, StudyPlanResponse
from src.services.study_planner import StudyPlannerService
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from src.models.database import get_session
from src.services.user_service import UserService
from src.services.study_session_service import StudySessionService
from src.utils.asrs_advisor import generate_advice

from src.utils.prediction import do_prediction, predict_new_sample

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Study Session Planner API"}


@router.get("/healthz")
async def health_check():
    return {"status": "ok"}


@router.post("/assess", response_model=StudyPlanResponse)
async def assess_study_plan(request: StudyPlanRequest):
    try:
        planner = StudyPlannerService()
        study_plan = planner.generate_plan(request)

        return StudyPlanResponse(
            participant_id=str(uuid.uuid4()),
            data=study_plan
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/users/")
def create_user(user_data: dict, session: Session = Depends(get_session)):
    user_service = UserService(session)
    return user_service.create_user(user_data)


@router.get("/users/{user_id}")
def get_user(user_id: int, session: Session = Depends(get_session)):
    user_service = UserService(session)
    return user_service.get_user(user_id)


@router.post("/study-sessions/")
def create_study_session(
        user_id: int,
        request: StudyPlanRequest,
        session: Session = Depends(get_session)
):
    study_service = StudySessionService(session)
    return study_service.create_session(
        user_id=user_id,
        duration=request.study_duration,
        break_duration=request.break_duration
    )


@router.get("/study-sessions/{user_id}")
def get_user_sessions(user_id: int, session: Session = Depends(get_session)):
    study_service = StudySessionService(session)
    return study_service.get_user_sessions(user_id)


@router.api_route("/api/predict/adhd-diagnosis", methods=["POST", "PUT"])
async def process_data(input_data: InputData):
    """
    处理输入数据
    :param input_data:
    :return:
    """
    # 将输入数据转换为字典
    data_dict = input_data.model_dump()

    # 定义所需的列
    columns = [
        'SEX', 'AGE', 'ACC', 'ACC_DAYS', 'HRV', 'HRV_HOURS'
        'CPT_II', 'ADD', 'BIPOLAR', 'UNIPOLAR',
        'ANXIETY', 'SUBSTANCE', 'OTHER', 'CT', 'MDQ_POS', 'WURS',
        'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MED', 'MED_Antidepr',
        'MED_Moodstab', 'MED_Antipsych', 'MED_Anxiety_Benzo', 'MED_Sleep',
        'MED_Analgesics_Opioids', 'MED_Stimulants', 'filter_$'
    ]

    # 过滤数据，只保留需要的列
    filtered_data = {k: data_dict.get(k) for k in columns}

    # 创建 DataFrame
    df = pd.DataFrame([filtered_data], columns=columns)

    res = await do_prediction(df)

    # instance to list/dict
    if isinstance(res, tuple):
        res = list(res)
    return {
        "message": "success",
        "prediction": {
            "negative_probability": res[0],
            "positive_probability": res[1]
        }
    }


@router.api_route("/api/ai/adhd-suggestions", methods=["POST", "PUT"])
async def process_data(request: Request):
    """
    处理输入数据
    :param request:
    :return:
    """
    try:
        result = await request.json()
    except Exception as e:
        print(f"Failed to parse input data: {str(e)}")
        raise HTTPException(status_code=400, detail="Input data must be a string")
    result = result.get("result")
    res = await generate_advice(result)
    return {
        "message": "success",
        "suggestions": res
    }