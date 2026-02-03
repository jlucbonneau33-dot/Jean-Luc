import logging
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key_env = os.getenv("OPENAI_API_KEY")
if not api_key_env:
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file")

client = OpenAI(api_key=api_key_env)

app = FastAPI()


def _error_payload(message: str, error_type: str = "invalid_request_error", code: str | None = None) -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(detail),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_payload("Invalid request"),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=_error_payload("Internal server error", error_type="server_error"),
    )

log_file = os.getenv("LOG_FILE", "logs/nova_api.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nova-api")

REDIS_URL = os.getenv("REDIS_URL")
_redis = None
if REDIS_URL:
    try:
        import redis  # type: ignore

        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
    except Exception as exc:
        logger.warning("Redis unavailable, falling back to in-memory rate limit: %s", exc)
        _redis = None

def _load_valid_keys() -> dict[str, str]:
    """
    Load API keys from environment.

    Supported formats:
    - NOVA_API_KEYS="key1:Name One,key2:Name Two"
    - NOVA_API_KEY="single-key" (name defaults to "user")
    """
    raw = os.getenv("NOVA_API_KEYS", "").strip()
    if raw:
        keys: dict[str, str] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                key, name = item.split(":", 1)
                keys[key.strip()] = name.strip() or "user"
            else:
                keys[item] = "user"
        return keys

    single = os.getenv("NOVA_API_KEY")
    if single:
        return {single: os.getenv("NOVA_USER_NAME", "user")}

    return {}


# You will give each paying user their own key
VALID_KEYS = _load_valid_keys()

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MAX_MESSAGES = int(os.getenv("NOVA_MAX_MESSAGES", "40"))
MAX_CONTENT_CHARS = int(os.getenv("NOVA_MAX_CONTENT_CHARS", "4000"))
REQUEST_TIMEOUT = int(os.getenv("NOVA_TIMEOUT", "60"))
RETRY_ATTEMPTS = int(os.getenv("NOVA_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.getenv("NOVA_RETRY_BACKOFF", "0.5"))
ALLOWED_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
}

_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)


def _rate_limit_key(api_key: str, minute: int) -> str:
    return f"nova:rl:{api_key}:{minute}"


def _rate_limit_allowed(api_key: str) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True

    if _redis is not None:
        minute = int(time.time() // 60)
        key = _rate_limit_key(api_key, minute)
        count = _redis.incr(key)
        if count == 1:
            _redis.expire(key, 61)
        return count <= RATE_LIMIT_PER_MINUTE

    now = time.time()
    window = 60.0
    bucket = _rate_buckets[api_key]
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return False
    bucket.append(now)
    return True


def _normalize_chat_response(payload: dict, model: str) -> dict:
    payload.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
    payload.setdefault("object", "chat.completion")
    payload.setdefault("created", int(time.time()))
    payload.setdefault("model", model)
    payload.setdefault("choices", [])
    payload.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return payload


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key", "anonymous")
    if VALID_KEYS:
        if api_key not in VALID_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if not _rate_limit_allowed(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    response = await call_next(request)
    return response


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/heartbeat")
async def heartbeat():
    return {"status": "ok", "time": time.time()}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": model_id, "object": "model"}
            for model_id in sorted(ALLOWED_MODELS)
        ]
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, http_request: Request):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="Messages are required")
    if len(request.messages) > MAX_MESSAGES:
        raise HTTPException(status_code=400, detail="Too many messages")

    if request.model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    for msg in messages:
        if not msg.get("content"):
            raise HTTPException(status_code=400, detail="Message content is required")
        if len(msg["content"]) > MAX_CONTENT_CHARS:
            raise HTTPException(status_code=400, detail="Message content too long")

    api_key = http_request.headers.get("X-API-Key", "anonymous")
    prompt_size = sum(len(msg.get("content", "")) for msg in messages)

    last_error: Exception | None = None
    start_time = time.perf_counter()
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=False,
                timeout=REQUEST_TIMEOUT,
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            response_text = response.choices[0].message.content or ""
            response_size = len(response_text)
            logger.info(
                "chat_completions ok api_key=%s model=%s prompt_chars=%s response_chars=%s duration_ms=%s",
                api_key,
                request.model,
                prompt_size,
                response_size,
                duration_ms,
            )
            return _normalize_chat_response(response.model_dump(), request.model)
        except Exception as exc:
            last_error = exc
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(
                "chat_completions error api_key=%s model=%s attempt=%s duration_ms=%s err=%s",
                api_key,
                request.model,
                attempt,
                duration_ms,
                exc,
            )
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            break

    raise HTTPException(status_code=502, detail="Upstream model error")


