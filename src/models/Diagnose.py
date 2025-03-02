# 定义输入数据模型
from pydantic import BaseModel, Field

class InputData(BaseModel):
    ID: str = Field(alias='id')
    SEX: int
    AGE: float
    ACC: str
    ACC_DAYS: int = Field(default=0)
    HRV_HOURS: float
    CPT_II: int
    ADD: int
    BIPOLAR: int
    UNIPOLAR: int
    ANXIETY: int
    SUBSTANCE: int
    OTHER: int
    CT: int
    MDQ_POS: int
    WURS: float
    ASRS: float
    MADRS: float
    HADS_A: float
    HADS_D: float
    MED: str
    MED_Antidepr: str
    MED_Moodstab: str
    MED_Antipsych: str
    MED_Anxiety_Benzo: str
    MED_Sleep: str
    MED_Analgesics_Opioids: str
    MED_Stimulants: str
    # 添加字段别名 filter_$，以避免关键字冲突
    filter_value: int = Field(default=0, alias='filter_$')

    # 处理额外的字段（可选）
    HRV: str = None
    HRV_TIME: str = None
    ADHD: int = None

    class Config:
        populate_by_name = True  # 允许使用字段别名作为输入键名

