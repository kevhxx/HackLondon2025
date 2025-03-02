# Project Overview

This project provides a FastAPI-based solution for ADHD-related data processing and study plan generation. Key functionalities include:

## Features

- **Data Preparation & Modeling**
    - A custom pipeline using SMOTE and `HistGradientBoostingClassifier` for ADHD diagnosis.
    - Utility methods for training, saving, and loading models.
    - Handling of specific data columns (e.g., *ID*, *SEX*, *AGE*, *ACC*, etc.) for prediction.

- **ASRS Advisor**
    - An asynchronous advisor function (`generate_advice`) for generating study recommendations based on ASRS test results.
    - Utilizes structured output (JSON) for coherent suggestions and study plan advice.

- **API Endpoints** (`src/api/routes.py`)
    - **Root** and **Health Check**: Basic check to verify service status.
    - **Assess Study Plan**: Accepts input, generates a customized study plan via `StudyPlannerService`.
    - **User Management**: Create and retrieve users with `UserService`.
    - **Study Sessions**: Create and retrieve study sessions using `StudySessionService`.
    - **Predict ADHD Diagnosis**: Receives input data, filters needed columns, and returns prediction results.
    - **AI ADHD Suggestions**: Accepts user input and generates ADHD-related advice.

- **Data Models**
    - **`InputData`**: Defines the expected structure for prediction inputs.
    - **`StudyPlanRequest` & `StudyPlanResponse`**: Models for handling study plan requests and responses.
    - **Database Integration**: `sqlmodel`-based models for users and study sessions.

- **File Structure**
    - **`src/models`**: Data model definitions.
    - **`src/services`**: Business logic (e.g., user, study planner, and study session services).
    - **`src/utils`**: Additional utilities such as prediction helpers and the ASRS advisor.
    - **`src/api/routes.py`**: FastAPI routes for handling API requests.

## Getting Started

**Install Dependencies**
   ```bash
   pip install -r requirements.txt
   uvicorn src.main:app --reload    
    ```
   