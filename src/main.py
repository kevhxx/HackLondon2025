import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
import redis
import os
from dotenv import load_dotenv

from src.api.routes import router
from src.middleware.correlation import CorrelationMiddleware
from src.middleware.timing import TimingMiddleware
from src.models.database import init_db
from fastapi.responses import JSONResponse

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    整个 FastAPI 生命周期的上下文管理器
    :param _: FastAPI 实例
    :return: None
    :param _:
    :return:
    """
    try:
        init_db()
        yield
    finally:
        pass


app = FastAPI(
    title="Study Session Planner API",
    description="Backend service for adaptive study planning",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET"))
app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationMiddleware)
app.add_middleware(TimingMiddleware)


@app.api_route("/api/status", methods=["GET", "POST"])
async def status_check(request: Request):
    """
    API 状态检查
    :param request:
    :return:
    """
    version_suffix = os.getenv("COMMIT_ID", "")[:8]
    if request.headers.get("Cf-Ray"):
        via = "Cloudflare"
        rayId = request.headers.get("Cf-Ray")
        realIp = request.headers.get("Cf-Connecting-Ip")
        dataCenter = request.headers.get("Cf-Ipcountry")
    elif request.headers.get("Eagleeye-Traceid"):
        via = "Aliyun"
        rayId = request.headers.get("Eagleeye-Traceid")
        realIp = request.headers.get("X-Real-Ip")
        dataCenter = request.headers.get("Via")
    else:
        return JSONResponse(content={"status": "error", "error": "Direct access not allowed"}, status_code=403)
    info = {
        "version": "v2.2-prod-" + version_suffix,
        "buildAt": os.environ.get("BUILD_AT", ""),
        "author": "binaryYuki <noreply.tzpro.xyz>",
        "arch": subprocess.run(['uname', '-m'], stdout=subprocess.PIPE).stdout.decode().strip(),
        "commit": os.getenv("COMMIT_ID", "")[:8],
        "instance-id": subprocess.run(['hostname'], stdout=subprocess.PIPE).stdout.decode().strip(),
        "request-id": request.headers.get("x-request-id", ""),
        "ray-id": rayId,
        "protocol": request.headers.get("X-Forwarded-Proto", ""),
        "ip": realIp,
        "dataCenter": dataCenter,
        "via": via,
        "code": 200,
        "message": "OK"
    }
    return JSONResponse(content=info, status_code=200)


# Include routers
app.include_router(router)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", ""),
    decode_responses=True
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
